#!/usr/bin/env python3

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import torch
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -------------------------------
# Minimal data structures
# -------------------------------

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SamplingParams:
    max_tokens: int = 8
    temperature: float = 0.0  # not used in this demo


class Sequence:
    _next_seq_id: int = 0

    def __init__(self, prompt_token_ids: List[int], sampling_params: SamplingParams):
        self.seq_id: int = Sequence._next_seq_id
        Sequence._next_seq_id += 1

        self.prompt_token_ids: List[int] = list(prompt_token_ids)
        self.completion_token_ids: List[int] = []

        self.sampling_params: SamplingParams = sampling_params
        self.status: SequenceStatus = SequenceStatus.WAITING

        # KV cache related metadata (host-side)
        self.block_table: List[int] = []  # list of block_ids used by this sequence
        self.num_cached_tokens: int = 0   # number of tokens whose KV is already in cache (via prefix sharing)

    def __len__(self) -> int:
        return len(self.prompt_token_ids) + len(self.completion_token_ids)

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.completion_token_ids

    @property
    def num_blocks(self) -> int:
        return (len(self) + self.block_size - 1) // self.block_size  # type: ignore[attr-defined]

    @property
    def last_block_num_tokens(self) -> int:
        remainder = len(self) % self.block_size  # type: ignore[attr-defined]
        return remainder if remainder != 0 else (self.block_size if len(self) > 0 else 0)

    def block(self, i: int) -> List[int]:
        assert hasattr(self, "block_size"), "block_size must be set on Sequence externally"
        start = i * self.block_size  # type: ignore[attr-defined]
        end = min(start + self.block_size, len(self))
        return self.all_token_ids[start:end]

    def append_token(self, token_id: int):
        self.completion_token_ids.append(token_id)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED


# -------------------------------
# Block and BlockManager (host-side metadata and allocation)
# -------------------------------

class Block:
    def __init__(self, block_id: int):
        self.block_id: int = block_id
        self.ref_count: int = 0
        self.hash: int = -1
        self.token_ids: List[int] = []

    def update(self, hash_value: int, token_ids: List[int]):
        self.hash = hash_value
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @staticmethod
    def compute_hash(token_ids: List[int], prefix: int = -1) -> int:
        # Simple rolling hash for demo (not collision-resistant)
        h = 1469598103934665603  # FNV offset basis
        if prefix != -1:
            h ^= prefix
            h *= 1099511628211
        for t in token_ids:
            h ^= (t & 0xFF_FF_FF_FF)
            h *= 1099511628211
            h &= (1 << 64) - 1
        return h

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1


# -------------------------------
# Scheduler
# -------------------------------

@dataclass
class Config:
    # KV cache parameters
    num_kvcache_blocks: int = 16
    kvcache_block_size: int = 8

    # Model parameters
    feature_dim: int = 16
    vocab_size: int = 128

    # Runtime constraints
    max_model_len: int = 128
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 128

    # Special tokens
    eos: int = 0


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.block_size = config.kvcache_block_size

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # attach block_size for block computations
        seq.block_size = self.block_size  # type: ignore[attr-defined]
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], bool]:
        # Prefill: bring waiting sequences into running, allocating blocks
        scheduled_seqs: List[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode: advance existing running sequences one token each
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs, "Scheduler must return something"
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) -> List[bool]:
        finished: List[bool] = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            is_done = (token_id == self.eos) or (len(seq.completion_token_ids) >= seq.sampling_params.max_tokens)
            if is_done:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
            finished.append(is_done)
        return finished


# -------------------------------
# ModelRunner (with minimal "model" using matrix multiplications)
# -------------------------------

class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.feature_dim = config.feature_dim
        self.vocab_size = config.vocab_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Pre-allocate a simple KV cache (no layers/heads in this minimal demo):
        # Shape: [num_blocks, block_size, feature_dim]
        self.kv_cache = torch.zeros(
            config.num_kvcache_blocks, self.block_size, self.feature_dim, dtype=torch.float32, device=self.device
        )

        # Simple token embedding and projection matrices
        torch.manual_seed(0)
        self.embeddings = torch.randn(self.vocab_size, self.feature_dim, device=self.device)
        self.projection = torch.randn(self.feature_dim, self.vocab_size, device=self.device)
        self.use_triton = TRITON_AVAILABLE and self.device.type == "cuda"

    # Triton kernel: copy rows from src (N, D) to dst (M, D) at indices given by index (N,)
    # We flatten src/dst for pointer arithmetic; each program handles one row.
    if TRITON_AVAILABLE:
        @staticmethod
        @triton.jit
        def _write_rows_kernel(src_ptr, dst_ptr, index_ptr, num_rows, FEATURE_DIM: tl.constexpr):
            pid = tl.program_id(0)
            if pid >= num_rows:
                return
            cols = tl.arange(0, FEATURE_DIM)
            row_idx = tl.load(index_ptr + pid)
            src_off = pid * FEATURE_DIM + cols
            dst_off = row_idx * FEATURE_DIM + cols
            vals = tl.load(src_ptr + src_off)
            tl.store(dst_ptr + dst_off, vals)

        @staticmethod
        @triton.jit
        def _attn_argmax_kernel(kv_ptr, indices_ptr, lengths_ptr, q_ptr, out_pos_ptr, BATCH: tl.constexpr, LMAX: tl.constexpr, FEATURE_DIM: tl.constexpr):
            pid = tl.program_id(0)  # sequence id
            if pid >= BATCH:
                return
            # load query q for this sequence
            cols = tl.arange(0, FEATURE_DIM)
            q = tl.load(q_ptr + pid * FEATURE_DIM + cols)
            length = tl.load(lengths_ptr + pid)
            max_val = tl.full((), -1e9, tl.float32)
            max_idx = tl.zeros((), tl.int32)
            # iterate over context positions
            for j in range(0, LMAX):
                valid = j < length
                idx = tl.load(indices_ptr + pid * LMAX + j)
                # guard pointer when invalid
                safe_idx = tl.where(valid, idx, 0)
                base = safe_idx * FEATURE_DIM
                k = tl.load(kv_ptr + base + cols, mask=valid, other=0.0)
                dot = tl.sum(k * q, axis=0)
                # set score to -inf for invalid positions
                score = tl.where(valid, dot, tl.full((), -1e9, tl.float32))
                better = score > max_val
                max_val = tl.where(better, score, max_val)
                max_idx = tl.where(better, j, max_idx)
            tl.store(out_pos_ptr + pid, max_idx)

        @staticmethod
        @triton.jit
        def _matmul_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k in range(0, K, BLOCK_K):
                a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
                b_ptrs = b_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
                b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)

                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                acc += tl.dot(a, b)

            c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc, mask=c_mask)

    def _write_to_kv_cache(self, slot_mapping: List[int], token_ids: List[int]):
        if not slot_mapping:
            return
        # Flattened view over [num_blocks * block_size, feature_dim]
        flat = self.kv_cache.view(-1, self.feature_dim)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        embed = self.embeddings[token_tensor]  # [N, feature_dim]
        index = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
        if self.use_triton:
            src_ptr = embed.view(-1)
            dst_ptr = flat.view(-1)
            num_rows = embed.shape[0]
            grid = (num_rows,)
            ModelRunner._write_rows_kernel[grid](
                src_ptr, dst_ptr, index, num_rows, FEATURE_DIM=self.feature_dim
            )
        else:
            flat[index] = embed

    def _prepare_prefill(self, seqs: List[Sequence]) -> Tuple[List[int], List[int]]:
        # For each sequence, compute slot mapping for NEW tokens only (not cached)
        slot_mapping: List[int] = []
        new_token_ids: List[int] = []
        last_new_token_ids: List[int] = []
        for seq in seqs:
            new_tokens = seq.all_token_ids[seq.num_cached_tokens:]
            # Prepare slot mapping into blocks for these new tokens
            for i, tok in enumerate(new_tokens):
                # Map to global slot index
                block_index = seq.block_table[(seq.num_cached_tokens + i) // self.block_size]
                offset_in_block = (seq.num_cached_tokens + i) % self.block_size
                global_slot = block_index * self.block_size + offset_in_block
                slot_mapping.append(global_slot)
                new_token_ids.append(tok)
            # The model will predict next token from the last token (like next-step LM)
            if new_tokens:
                last_new_token_ids.append(new_tokens[-1])
            else:
                # if fully cached prefix, use last token of sequence
                last_new_token_ids.append(seq.all_token_ids[-1])
        # Write embeddings of new tokens into KV cache
        self._write_to_kv_cache(slot_mapping, new_token_ids)
        return last_new_token_ids, slot_mapping

    def _prepare_decode(self, seqs: List[Sequence]) -> Tuple[List[int], List[int]]:
        # One-step decode: use last token as input; map its cache slot (end position)
        last_tokens: List[int] = []
        slot_mapping: List[int] = []
        for seq in seqs:
            last_tokens.append(seq.all_token_ids[-1])
            # slot of the last position in sequence
            last_pos = len(seq) - 1
            block_index = seq.block_table[last_pos // self.block_size]
            offset_in_block = last_pos % self.block_size
            slot_mapping.append(block_index * self.block_size + offset_in_block)
        return last_tokens, slot_mapping

    def _build_context_indices(self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build flat-slot indices for all positions in each sequence
        lengths = [len(seq) for seq in seqs]
        Lmax = max(lengths)
        B = len(seqs)
        indices = torch.full((B, Lmax), -1, dtype=torch.long, device=self.device)
        for i, seq in enumerate(seqs):
            for pos in range(len(seq)):
                block_index = seq.block_table[pos // self.block_size]
                offset_in_block = pos % self.block_size
                indices[i, pos] = block_index * self.block_size + offset_in_block
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=self.device)
        return indices, lengths_tensor

    @torch.inference_mode()
    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        # Prepare inputs and write KV for prefill
        if is_prefill:
            input_token_ids, _ = self._prepare_prefill(seqs)
        else:
            input_token_ids, _ = self._prepare_decode(seqs)

        # Minimal "LM head": logits = E[token] @ W; greedy argmax
        tokens = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
        hidden = self.embeddings[tokens]  # [B, D]
        # Optional: Triton attention-like retrieval from kv_cache to augment hidden
        if self.use_triton and len(seqs) > 0:
            kv_flat = self.kv_cache.view(-1, self.feature_dim)
            indices, lengths = self._build_context_indices(seqs)
            B, Lmax = indices.shape
            out_pos = torch.empty(B, dtype=torch.int32, device=self.device)
            ModelRunner._attn_argmax_kernel[(B,)](
                kv_flat.view(-1), indices.view(-1), lengths, hidden.view(-1), out_pos,
                BATCH=B, LMAX=Lmax, FEATURE_DIM=self.feature_dim,
            )
            # gather best context row per sequence
            gather_rows = indices[torch.arange(B, device=self.device), out_pos.long()]
            k_sel = kv_flat[gather_rows]  # [B, D]
            hidden = hidden + k_sel
        if self.use_triton:
            # Triton matmul: [B, D] x [D, V] -> [B, V]
            a = hidden.contiguous()
            b = self.projection.contiguous()
            B, D = a.shape
            D2, V = b.shape
            assert D == D2
            logits = torch.empty((B, V), device=self.device, dtype=a.dtype)
            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32
            grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(V, BLOCK_N))
            ModelRunner._matmul_kernel[grid](
                a, b, logits,
                B, V, D,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                logits.stride(0), logits.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
        else:
            logits = hidden @ self.projection  # [B, V]
        next_token_ids = torch.argmax(logits, dim=-1).tolist()
        return next_token_ids


# -------------------------------
# Demo main
# -------------------------------

def main():
    # Configure a tiny setup
    config = Config(
        num_kvcache_blocks=16,
        kvcache_block_size=8,
        feature_dim=16,
        vocab_size=128,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        eos=0,
    )

    # Build runtime pieces
    scheduler = Scheduler(config)
    runner = ModelRunner(config)

    # Create a few toy prompts as token id lists (non-zero so they don't instantly EOS)
    prompts = [
        [10, 11, 12, 13, 14, 15],
        [20, 21, 22, 23],
    ]
    sampling = [SamplingParams(max_tokens=6), SamplingParams(max_tokens=6)]

    seqs = [Sequence(p, sp) for p, sp in zip(prompts, sampling)]
    for seq in seqs:
        scheduler.add(seq)

    step_idx = 0
    print("Starting generation...\n")
    while not scheduler.is_finished():
        step_idx += 1
        batch, is_prefill = scheduler.schedule()
        next_tokens = runner.run(batch, is_prefill)
        finished_flags = scheduler.postprocess(batch, next_tokens)

        # Pretty print step info
        stage = "PREFILL" if is_prefill else "DECODE"
        print(f"Step {step_idx:02d} [{stage}] -> next tokens: {next_tokens} | finished: {finished_flags}")
        for b in batch:
            print(f"  Seq {b.seq_id} len={len(b)} blocks={b.block_table} cached={b.num_cached_tokens}")
        print()

    print("All sequences finished.\n")
    for s in seqs:
        print(f"Seq {s.seq_id} prompt={s.prompt_token_ids}")
        print(f" -> completion={s.completion_token_ids}")
        print()


if __name__ == "__main__":
    # Make CPU deterministic for repeatability
    random.seed(0)
    torch.manual_seed(0)
    main()
