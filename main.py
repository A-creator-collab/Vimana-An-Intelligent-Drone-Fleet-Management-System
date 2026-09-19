"""main.py - End-to-end demo of the Intelligent Drone Fleet Management System.

Simulation-only. Pure Python + numpy + scikit-learn + matplotlib. CPU only, ~10-20 s on Colab.
Sections: data -> DSA showcase -> ML models -> NLP commands -> simulation
(Hungarian vs greedy) -> fleet analysis -> benchmarks -> plots -> final report.
"""
import time

import numpy as np

import benchmark
from ai.anomaly_detection import run_stream_detection, train_isolation_forest
from ai.demand_forecasting import allocate_drones, forecast_next_hour, train_forecaster
from ai.nlp_command_parser import NLPCommandParser, CommandParseError
from ai.predictive_maintenance import predict_fleet, train_maintenance_model
from dashboard import plots, report
from data_generator import (ZONES, make_city, make_demand_history, make_fleet, make_maintenance_dataset,
                            make_no_fly_events, make_task_stream, make_telemetry_stream, set_seeds)
from dsa.assignment import greedy_assign, hungarian
from dsa.dp_routing import brute_force_tsp, constrained_route, held_karp, nearest_neighbor_tsp
from dsa.extras import Trie, merge_sort
from dsa.pathfinding import astar, dijkstra, floyd_warshall, fw_path
from dsa.scheduling import AVLTree, ReservationTable, TaskQueue
from dsa.spatial_index import KDTree
from dsa.union_find import UnionFind
from simulation_engine import SimulationEngine

N_DRONES, N_TASKS, HORIZON, TICKS, N_NOFLY, SEED = 24, 110, 120, 260, 3, 42
T0 = time.time()


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def build_scenario(strategy, grounded=()):
    """Fresh, identical scenario for every run (fixed seeds) so strategies are comparable."""
    g, hubs = make_city()
    apsp, _ = floyd_warshall(g)
    sim = SimulationEngine(g, make_fleet(N_DRONES, g, hubs), make_task_stream(N_TASKS, HORIZON, g), hubs,
                           make_no_fly_events(N_NOFLY, HORIZON, g), apsp, strategy)
    sim.ground(grounded)
    sim.schedule_command(20, "send 2 drones to sector B and return when battery hits 20%")
    sim.schedule_command(30, "deliver a 2 kg package from node 5 to node 88 priority 5")
    sim.schedule_command(45, "recall drone 3")
    sim.schedule_command(50, "what is the status of drone 3")
    sim.schedule_command(55, "send 99 drones to sector A")
    sim.schedule_command(60, "close sector D for 15 ticks")
    sim.schedule_command(61, "make me a sandwich")
    return sim


def main():
    set_seeds(SEED)
    R = {}   # report context - everything goes in here

    # ------------------------------------------------------------------ 1. data
    banner("1. SYNTHETIC DATA (seed 42)")
    g, hubs = make_city()
    apsp, nxt = floyd_warshall(g)
    n_corr = sum(len(a) for a in g.adj.values()) // 2
    print(f"City graph : {g.n} waypoints, {n_corr} corridors, hubs={hubs}")
    print(f"Fleet      : {N_DRONES} drones | Tasks: {N_TASKS} over {HORIZON} ticks | No-fly events: {N_NOFLY}")
    print("Floyd-Warshall all-pairs table built for", g.n, "nodes (O(1) dispatch cost lookup)")
    R["city"] = {"nodes": g.n, "corridors": n_corr, "hubs": hubs}

    # ------------------------------------------------------------------ 2. DSA
    banner("2. DSA SHOWCASE")
    a, b = 0, 99
    pd_, cd, ed = dijkstra(g, a, b); pa, ca, ea = astar(g, a, b)
    same = abs(cd - ca) < 1e-9 and abs(cd - apsp[a][b]) < 1e-9
    print(f"Dijkstra 0->99: cost {cd:.1f}, expanded {ed} | A*: cost {ca:.1f}, expanded {ea} | "
          f"FW cost {apsp[a][b]:.1f} | same optimum: {same}")
    print("A* path:", pa)
    reroute = astar(g, a, b, frozenset(pa[3:6]))[0]
    print("A* re-route with 3 nodes blocked:", reroute)
    kd = KDTree([(n.x, n.y, n.id) for n in g.nodes.values()])
    kd3 = [(round(d, 1), i) for d, i in kd.nearest(33, 47, 3)]
    print("KD-tree 3 nearest waypoints to (33, 47):", kd3)
    cost = [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]]
    h_toy, g_toy = hungarian(cost), greedy_assign(cost)
    print("Hungarian on toy 4x4:", h_toy, "| greedy:", g_toy)
    stops = [0, 12, 37, 58, 71, 94]
    dm = [[apsp[u][v] for v in stops] for u in stops]
    hk, order = held_karp(dm); nn, _ = nearest_neighbor_tsp(dm); bf, _ = brute_force_tsp(dm)
    order_nodes = [stops[i] for i in order]
    print(f"Multi-stop route (DP): {order_nodes} cost {hk:.1f} | brute force {bf:.1f} | nearest-neighbour {nn:.1f}")
    cc, co, ex = constrained_route(dm, max_distance=hk * 1.1, speed=5.0, deadlines=[0] + [hk / 5 * 1.1] * 5)
    co_nodes = [stops[i] for i in co] if co else None
    print(f"Backtracking + B&B (battery & deadline constrained): cost {cc:.1f}, order {co_nodes}, explored {ex} states")
    q = TaskQueue()
    for t in make_task_stream(6, 50, g, seed=1): q.push(t)
    q_order = [(t.priority, t.deadline) for t in q.pop_many(6)]
    print("Heap task queue pop order (priority, deadline):", q_order)
    avl = AVLTree()
    for i, bat in enumerate([88, 45, 97, 12, 63, 71, 30, 55]): avl.insert((bat, i), i)
    avl_range = [k for k, _ in avl.range_query((60, -1))]
    print(f"AVL (battery,id): height {avl.height()} for {len(avl)} keys; drones >= 60%:", avl_range)
    rt = ReservationTable(capacity=1)
    rt.reserve("corridor-7-8", 10, 14)
    r12 = rt.is_free("corridor-7-8", 12, 13); r15 = rt.is_free("corridor-7-8", 15, 17)
    print("Reservation table: corridor free at t=12?", r12, "| free at t=15?", r15)
    uf = UnionFind(6); uf.union(0, 1); uf.union(1, 2); uf.union(4, 5)
    uf_groups = uf.groups()
    print("Union-Find groups:", uf_groups)
    trie = Trie()
    for w in ["drone-01", "drone-02", "depot-north", "delta-hub", "hub-a"]: trie.insert(w)
    trie_d = trie.starts_with("d")
    ms = merge_sort([5, 2, 9, 1], reverse=True)
    print("Trie prefix 'd' ->", trie_d, "| merge sort by (-x):", ms)
    R["dsa"] = {
        "dijkstra_cost": cd, "dijkstra_expanded": ed,
        "astar_cost": ca, "astar_expanded": ea,
        "fw_cost": apsp[a][b], "optimum_match": same,
        "astar_path": pa, "astar_reroute": reroute,
        "kd_3nn": kd3,
        "hungarian_toy": h_toy, "greedy_toy": g_toy,
        "tsp_dp_cost": hk, "tsp_order": order_nodes,
        "tsp_brute_cost": bf, "tsp_nn_cost": nn,
        "constrained_cost": cc, "constrained_order": co_nodes, "constrained_explored": ex,
        "task_queue": q_order,
        "avl_height": avl.height(), "avl_range": avl_range,
        "resv_t12": r12, "resv_t15": r15,
        "uf_groups": uf_groups,
        "trie_d": trie_d, "merge_sort": ms,
    }

    # ------------------------------------------------------------------ 3. ML
    banner("3. MACHINE LEARNING (scikit-learn, CPU, seconds)")
    t = time.time()
    X, y = make_maintenance_dataset()
    m_model, m_res = train_maintenance_model(X, y)
    print(f"Predictive maintenance (RandomForest, 3000 synthetic flights): accuracy {m_res['accuracy']:.3f}, F1 {m_res['f1']:.3f}")
    print("  feature importances:", m_res["importances"])
    fleet = make_fleet(N_DRONES, g, hubs)
    risk = predict_fleet(m_model, fleet)
    grounded = sorted(i for i, p in risk.items() if p > 0.85)
    print(f"  fleet risk: {len(grounded)} drone(s) with P(maintenance) > 0.85 -> GROUNDED before simulation: {grounded}")
    tel = make_telemetry_stream()
    iso, prec, rec = train_isolation_forest(tel)
    sp, sr = run_stream_detection(tel)
    print(f"Anomaly detection: IsolationForest precision {prec:.3f} recall {rec:.3f} | sliding-window z-score (drain channel) precision {sp:.3f} recall {sr:.3f}")
    counts = make_demand_history()
    fmodel, fres = train_forecaster(counts)
    print(f"Demand forecast (LinearRegression on lags): MAE {fres['mae']:.2f}, RMSE {fres['rmse']:.2f} "
          f"| naive lag-1 MAE {fres['naive_lag1_mae']:.2f}, lag-24 MAE {fres['naive_lag24_mae']:.2f}")
    pred = forecast_next_hour(fmodel, counts)
    next_demand = dict(zip(ZONES, map(float, pred.round(1))))
    alloc = dict(zip(ZONES, allocate_drones(pred, N_DRONES)))
    print("  next-hour predicted demand per zone:", next_demand, "-> pre-position drones:", alloc)
    print(f"  (ML section took {time.time() - t:.2f}s)")
    R["ml"] = {
        "maint_accuracy": m_res["accuracy"], "maint_f1": m_res["f1"],
        "maint_importances": m_res["importances"],
        "grounded": grounded,
        "iso_precision": prec, "iso_recall": rec,
        "sw_precision": sp, "sw_recall": sr,
        "fc_mae": fres["mae"], "fc_rmse": fres["rmse"],
        "fc_naive1": fres["naive_lag1_mae"], "fc_naive24": fres["naive_lag24_mae"],
        "fc_next": next_demand, "fc_alloc": alloc,
    }

    # ------------------------------------------------------------------ 4. NLP
    banner("4. NLP COMMAND PARSER (regex rules) -> structured commands")
    parser = NLPCommandParser()
    nlp_log = []
    for s in ["Send 2 drones to sector B and return when battery hits 20%", "deliver 3.5 kg from sector A to node 77 priority 4",
              "inspect sector C with three drones", "recall all drones", "block sector D for 30 ticks", "juggle the drones"]:
        try:
            c = parser.parse(s)
            out = f"{c.action} {c.params}"
        except CommandParseError as e:
            out = f"PARSE ERROR ({e})"
        nlp_log.append((s, out))
        print(f"  '{s}'\n      -> {out}")
    R["nlp"] = nlp_log

    # ------------------------------------------------------------------ 5. simulation
    banner(f"5. SIMULATION: {TICKS} ticks, Hungarian vs Greedy dispatch (identical scenario, {len(grounded)} drones grounded)")
    sims = []
    for strat in ("hungarian", "greedy"):
        sim = build_scenario(strat, grounded)
        sim.run(TICKS)
        sims.append(sim)
    keys = ["tasks_total", "completed", "completion_rate_%", "late", "avg_turnaround_ticks", "avg_battery_reserve_%",
            "fleet_distance", "distance_per_task", "validator_rejections", "replans", "hold_ticks", "crashed",
            "proximity_events", "astar_nodes_expanded", "dispatch_time_s"]
    r1, r2 = sims[0].report(), sims[1].report()
    print(f"{'metric':<26}{'hungarian':>12}{'greedy':>12}")
    for k in keys:
        print(f"{k:<26}{r1[k]:>12}{r2[k]:>12}")
    print("rejection breakdown (hungarian):", r1["rejection_breakdown"])
    print("\nNLP commands executed during the hungarian run (all pass through the safety validator):")
    for line in sims[0].command_log:
        print("  ", line)
    R["sims"] = sims

    banner("6. FLEET ANALYSIS at end of run (Union-Find swarms, graph-colouring channels)")
    an = sims[0].analyze_fleet()
    print("swarm clusters (Union-Find, radius 25):", [c for c in an["swarm_clusters"]])
    print("radio channels needed (Welsh-Powell colouring):", an["channels_used"], "| merged no-fly regions:", an["no_fly_regions"])
    print("top drones (id, tasks done, battery):", an["top_drones"])
    R["fleet"] = an

    # ------------------------------------------------------------------ 7. benchmarks
    banner("7. BENCHMARKS (Big-O verified empirically)")
    bres = benchmark.run_all()
    R["benchmarks"] = bres

    # ------------------------------------------------------------------ 8. plots
    banner("8. PLOTS")
    iso_flags = (iso.predict(np.column_stack([tel["drain"], tel["off_course"], tel["temp"]])) == -1).astype(int)
    files = [plots.plot_city(sims[0]), plots.plot_sim_compare(sims), plots.plot_benchmarks(bres),
             plots.plot_ml(m_res["importances"], fres, tel, iso_flags)]
    print("saved:", files)
    R["plot_files"] = files

    # ------------------------------------------------------------------ 9. final report
    banner("9. FINAL REPORT")
    R.update(runtime_s=time.time() - T0, seed=SEED, ticks=TICKS,
             n_drones=N_DRONES, n_tasks=N_TASKS, horizon=HORIZON, n_no_fly=N_NOFLY)
    report_files = report.write_report(R)
    print("report written:", report_files)
    print(f"\nTOTAL RUNTIME: {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()