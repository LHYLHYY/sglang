"""CPU contracts for Ascend asymmetric MLA prefill/decode KV transfer.

Run directly with Python and NumPy; no torch, NPU runtime or transfer service is
required. Unchanged production methods build the rank mappings and byte ranges.
A local memory-copy stub verifies the payloads actually reach their consumers.
"""

import ast
import concurrent.futures
import copy
import ctypes
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def load_definitions(path, names, namespace, class_name=None):
    source_path = SRT / path
    source = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = source.body
    if class_name:
        nodes = next(
            node.body
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    definitions = [
        copy.deepcopy(node)
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != set(names):
        raise AssertionError(
            f"Missing production definitions in {source_path}: {names}"
        )
    if class_name:
        definitions = [
            ast.ClassDef(
                name=class_name,
                bases=[],
                keywords=[],
                body=definitions,
                decorator_list=[],
            )
        ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace
    )
    return namespace


class TestAsymmetricMLARankMapping(unittest.TestCase):
    def setUp(self):
        self.namespace = {"logger": Mock()}
        load_definitions(
            "disaggregation/common/conn.py",
            ["_resolve_rank_mapping"],
            self.namespace,
            class_name="CommonKVManager",
        )
        load_definitions(
            "disaggregation/common/conn.py",
            ["_setup_bootstrap_infos"],
            self.namespace,
            class_name="CommonKVReceiver",
        )

    def resolve(self, engine_rank, prefill_tp):
        manager = self.namespace["CommonKVManager"]()
        manager.attn_tp_size = 2
        manager.attn_cp_size = manager.pp_size = 1
        manager.attn_cp_rank = manager.pp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_all_cp_ranks_for_transfer = False
        manager.kv_args = SimpleNamespace(engine_rank=engine_rank)
        info = SimpleNamespace(
            attn_tp_size=prefill_tp,
            attn_cp_size=1,
            pp_size=1,
            enable_dsa_cache_layer_split=False,
        )
        manager._resolve_rank_mapping(info)
        return manager, info

    def bootstrap(self, manager, info, prefill_dp_rank=0):
        manager.connection_pool = {}
        receiver = self.namespace["CommonKVReceiver"]()
        receiver.kv_mgr = manager
        receiver.bootstrap_addr = "prefill:8998"
        receiver.prefill_dp_rank = prefill_dp_rank
        receiver.__dict__.update(info.__dict__)
        receiver._get_bootstrap_info_from_server = Mock(
            side_effect=lambda dp, cp, tp, pp: {
                "dp_rank": dp,
                "cp_rank": cp,
                "tp_rank": tp,
                "pp_rank": pp,
            }
        )
        receiver._register_kv_args = Mock()
        receiver._setup_bootstrap_infos()
        return receiver

    def test_equal_tp2_keeps_indexer_and_compute_roles_for_all_engine_ranks(self):
        for engine_rank in range(16):
            with self.subTest(engine_rank=engine_rank):
                manager, info = self.resolve(engine_rank, prefill_tp=2)
                role = engine_rank % 2
                self.assertEqual(info.target_tp_rank, role)
                self.assertEqual(info.target_tp_ranks, [role])
                self.assertEqual(info.required_dst_info_num, 1)
                self.assertEqual(info.required_prefill_response_num, 1)
                receiver = self.bootstrap(manager, info, prefill_dp_rank=7)
                self.assertEqual(
                    [
                        (entry["dp_rank"], entry["tp_rank"], entry["is_dummy"])
                        for entry in receiver.bootstrap_infos
                    ],
                    [(7, role, False)],
                )

    def test_symmetric_prefill_maps_to_both_decode_roles_and_counts_dummy_peers(self):
        for prefill_tp in (1, 2, 4, 8):
            real_prefill_ranks = []
            for role in (0, 1):
                with self.subTest(prefill_tp=prefill_tp, role=role):
                    manager, info = self.resolve(role, prefill_tp)
                    receiver = self.bootstrap(manager, info)
                    entries = receiver.bootstrap_infos
                    real = [entry for entry in entries if not entry["is_dummy"]]
                    dummy = [entry for entry in entries if entry["is_dummy"]]
                    self.assertEqual(len(real), 1)
                    self.assertEqual(info.required_prefill_response_num, 1)
                    self.assertEqual(
                        info.required_dst_info_num, 2 if prefill_tp == 1 else 1
                    )
                    self.assertEqual(len(dummy), max(prefill_tp // 2 - 1, 0))
                    expected = 0 if prefill_tp == 1 else role * (prefill_tp // 2)
                    self.assertEqual(real[0]["tp_rank"], expected)
                    self.assertEqual(info.target_cp_ranks, [0])
                    self.assertEqual(info.target_pp_ranks, [0])
                    real_prefill_ranks.append(expected)
            self.assertEqual(len(set(real_prefill_ranks)), min(prefill_tp, 2))

    def test_reused_connection_preserves_dummy_assignments_without_reregistering(self):
        manager, info = self.resolve(1, prefill_tp=8)
        receiver = self.bootstrap(manager, info)
        first_entries = receiver.bootstrap_infos
        receiver._setup_bootstrap_infos()
        self.assertEqual(receiver.bootstrap_infos, first_entries)
        self.assertEqual(receiver._get_bootstrap_info_from_server.call_count, 4)
        receiver._register_kv_args.assert_called_once()


class Buffer:
    def __init__(self, array):
        self.array = array
        self.nbytes = array.nbytes

    def data_ptr(self):
        return self.array.ctypes.data

    def __getitem__(self, index):
        return self.array[index]


class TestAsymmetricMLAKVTransfer(unittest.TestCase):
    def setUp(self):
        self.namespace = {"np": np, "concurrent": concurrent, "logger": Mock()}
        load_definitions(
            "disaggregation/common/utils.py",
            ["group_concurrent_contiguous"],
            self.namespace,
        )
        load_definitions(
            "hardware_backend/npu/memory_pool_npu.py",
            ["get_contiguous_buf_infos"],
            self.namespace,
            class_name="NPUMLATokenToKVPool",
        )
        load_definitions(
            "disaggregation/ascend/conn.py",
            ["send_kvcache", "get_mla_kv_ptrs_with_pp"],
            self.namespace,
            class_name="AscendKVManager",
        )

    def pool(self, fill):
        pool = self.namespace["NPUMLATokenToKVPool"]()
        pool.layer_num = 3
        pool.index_head_dim = 128
        # Distinct group lengths catch accidentally treating all groups as KV.
        for group_id, (name, item_bytes) in enumerate(
            (("k_buffer", 128), ("v_buffer", 1024), ("index_k_buffer", 256))
        ):
            setattr(
                pool,
                name,
                [
                    Buffer(
                        np.full(
                            (10, item_bytes),
                            fill + 10 * group_id + layer,
                            dtype=np.uint8,
                        )
                        + np.arange(10, dtype=np.uint8)[:, None]
                    )
                    for layer in range(pool.layer_num)
                ],
            )
        return pool

    def transfer(self, source, destination, custom_pool=False, status=0):
        source_ptrs, source_lens, item_lens = source.get_contiguous_buf_infos()
        destination_ptrs, _, _ = destination.get_contiguous_buf_infos()
        manager = self.namespace["AscendKVManager"]()
        manager.pp_size = 1
        manager.is_mla_backend = True
        manager.enable_custom_mem_pool = custom_pool
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=source_ptrs, kv_data_lens=source_lens, kv_item_lens=item_lens
        )
        blocks = []

        def copy_payload(session, transfer_blocks):
            self.assertEqual(session, "decode-session")
            blocks.extend(transfer_blocks)
            if status:
                return status
            for source_address, destination_address, size in transfer_blocks:
                ctypes.memmove(destination_address, source_address, size)
            return 0

        manager._transfer_data = Mock(side_effect=copy_payload)
        source_indices = np.array([1, 2, 5, 6], dtype=np.int32)
        destination_indices = np.array([3, 4, 7, 9], dtype=np.int32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            result = manager.send_kvcache(
                "decode-session",
                source_indices,
                destination_ptrs,
                destination_indices,
                executor,
            )
        return result, manager, blocks

    def test_pool_exports_all_key_value_and_index_layers_in_group_order(self):
        pool = self.pool(10)
        ptrs, lengths, items = pool.get_contiguous_buf_infos()
        expected = pool.k_buffer + pool.v_buffer + pool.index_k_buffer
        self.assertEqual(ptrs, [buffer.data_ptr() for buffer in expected])
        self.assertEqual(lengths, [buffer.nbytes for buffer in expected])
        self.assertEqual(items, [128] * 3 + [1024] * 3 + [256] * 3)

    def test_transfer_copies_complete_pages_for_all_three_buffer_groups(self):
        for custom_pool in (False, True):
            with self.subTest(custom_pool=custom_pool):
                source, destination = self.pool(30), self.pool(0)
                before = {
                    name: [buffer.array.copy() for buffer in getattr(destination, name)]
                    for name in ("k_buffer", "v_buffer", "index_k_buffer")
                }
                result, manager, blocks = self.transfer(
                    source, destination, custom_pool
                )
                self.assertEqual(result, 0)
                self.assertEqual(len(blocks), 9 * 3)
                self.assertEqual(
                    manager._transfer_data.call_count, 9 if custom_pool else 1
                )
                for name in ("k_buffer", "v_buffer", "index_k_buffer"):
                    for src, dst, initial in zip(
                        getattr(source, name), getattr(destination, name), before[name]
                    ):
                        np.testing.assert_array_equal(
                            dst.array[[3, 4, 7, 9]], src.array[[1, 2, 5, 6]]
                        )
                        untouched = [0, 1, 2, 5, 6, 8]
                        np.testing.assert_array_equal(
                            dst.array[untouched], initial[untouched]
                        )
                        self.assertEqual(
                            sum(
                                length
                                for address, _, length in blocks
                                if src.data_ptr()
                                <= address
                                < src.data_ptr() + src.nbytes
                            ),
                            4 * src.array[0].nbytes,
                        )

    def test_symmetric_prefill_populates_each_decode_roles_consumed_cache(self):
        for role in (0, 1):
            source, destination = self.pool(40), self.pool(0)
            consumed = ("index_k_buffer",) if role == 0 else ("k_buffer", "v_buffer")
            result, _, _ = self.transfer(source, destination)
            self.assertEqual(result, 0)
            for name in consumed:
                for src, dst in zip(getattr(source, name), getattr(destination, name)):
                    np.testing.assert_array_equal(
                        dst.array[[3, 4, 7, 9]], src.array[[1, 2, 5, 6]]
                    )

    def test_transfer_error_is_propagated(self):
        for custom_pool in (False, True):
            result, _, _ = self.transfer(
                self.pool(30), self.pool(0), custom_pool, status=-7
            )
            self.assertEqual(result, -7)


if __name__ == "__main__":
    unittest.main()
