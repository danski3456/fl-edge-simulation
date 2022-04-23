# %%
from config import settings as st
import pyomo.environ as pyo
import networkx as nx
from plots import draw_exchange_graph


def get_graph_from_params():

    N = st.NUM_CLIENTS
    topology = st.OPTIMAL_DATA_SHARING.TOPOLOGY
    if topology == "ring":
        g = nx.cycle_graph(N)
    else:
        raise NotImplementedError

    for e in g.edges():
        g.edges[e]["capacity"] = st.OPTIMAL_DATA_SHARING.EDGE_CAPACITY

    return g

    


def get_abstrac_model():

    m = pyo.AbstractModel()


    # Sets of indices
    m.I = pyo.Set()  # set of clients
    m.C = pyo.Set()  # set of classes
    m.E = pyo.Set(within=m.I * m.I)  # edges of the graph


    def remove_self_commodities(model, i, j, c):
        return i != j


    m.K = pyo.Set(
        initialize=m.I * m.I * m.C, filter=remove_self_commodities
    )  # commodities specific exchanges


    m.NodesOut = pyo.Set(m.I, within=m.I)
    m.NodesIn = pyo.Set(m.I, within=m.I)


    def Populate_In_and_Out(m):
        # loop over the arcs and record the end points
        for i, j in m.E:
            m.NodesIn[j].add(i)
            m.NodesOut[i].add(j)


    m.In_n_Out = pyo.BuildAction(rule=Populate_In_and_Out)

    #  Variables
    m.exchange = pyo.Var(
        m.K, within=pyo.NonNegativeReals
    )  # the amount of items sent from one client to the other for a given class
    m.flows = pyo.Var(m.E, m.K)  # the flows in each edge or each specific exchange
    m.flows_aux = pyo.Var(m.E, m.K)

    # Parameters
    m.u = pyo.Param(m.I, m.C)  # initial assignament
    m.ec = pyo.Param(m.E)  # edge capacities


    # Constraints

    # Balance Flows
    def compute_flow_constraints(model, src, dest, cls, node):

        print(src, dest, cls)
        if node == src:
            rhs = model.exchange[src, dest, cls]
        elif node == dest:
            rhs = -model.exchange[src, dest, cls]
        else:
            rhs = 0

        lhs = sum(
            model.flows[(node, j, src, dest, cls)] for j in model.NodesOut[node]
        ) - sum(model.flows[(j, node, src, dest, cls)] for j in model.NodesIn[node])

        return lhs == rhs


    m.flow_const = pyo.Constraint(m.K * m.I, rule=compute_flow_constraints)


    def compute_capacity_constraint(model, src, dest, cls):
        return model.exchange[src, dest, cls] <= model.u[src, cls]


    m.capacity_const = pyo.Constraint(m.K, rule=compute_capacity_constraint)

    # relax E |f_k|  <= ec
    # and use
    # E t_k <= ec, t_k >= f_k, t_k >= -f_k
    def compute_edge_capacity_pos_constraint(model, src, dest):
        lhs = sum(model.flows_aux[src, dest, i, j, k] for (i, j, k) in model.K)
        return (0, lhs, model.ec[src, dest])


    m.edge_capacity_pos_const = pyo.Constraint(
        m.E, rule=compute_edge_capacity_pos_constraint
    )


    def compute_flow_aux_up_bounds(model, src, dest, i, j, c):
        k = (src, dest, i, j, c)
        return model.flows_aux[k] - model.flows[k] >= 0


    m.flow_aux_up_bounds = pyo.Constraint(m.E, m.K, rule=compute_flow_aux_up_bounds)


    def compute_flow_aux_low_bounds(model, src, dest, i, j, c):
        k = (src, dest, i, j, c)
        return model.flows[k] + model.flows_aux[k] >= 0


    m.flow_aux_low_bounds = pyo.Constraint(m.E, m.K, rule=compute_flow_aux_low_bounds)


    def objective(model):

        squares = []
        num_assertions_end_of_set = 0
        for i in model.I:
            for c in model.C:
                try:
                    # calculate the samples of class c
                    num_c = model.u[i, c] + sum(
                        model.exchange[j, i, c]
                        for j in model.I
                        if (j, i, c) in model.exchange
                    )
                    c_next = model.C.next(c)
                    # calculate the samples of another class c
                    num_c_next = model.u[i, c_next] + sum(
                        model.exchange[j, i, c_next]
                        for j in model.I
                        if (j, i, c_next) in model.exchange
                    )
                    term = (num_c - num_c_next) ** 2  # force both quantities to be similar
                    squares.append(term)
                except IndexError as e:
                    num_assertions_end_of_set += 1

        assert num_assertions_end_of_set == len(model.I)  # only one assertion per node

        return sum(squares)


    def objective_flow(model):
        return sum(model.exchange[k] for k in model.K)


    m.objective = pyo.Objective(rule=objective, sense=1)

    return m

def solve_instance(data_assignment: dict):

    network = get_graph_from_params()
    classes = list(range(st.NUM_CLASSES))

    data = {}
    data["I"] = list(network.nodes())
    data["C"] = classes 
    data["E"] = list(network.edges())
    # data["E"].extend([(v, u) for (u, v) in data["E"]])
    data["u"] = data_assignment
    # for i in data["I"]:
    #     for c in data["C"]:
    #         data["u"][(i, c)] = 1000 if (c % 6) == i else 0
    # data["ec"] = {}
    # for e in data["E"]:
    #     data["ec"][e] = 1000

    for u, v, d in g.edges(data=True):
        data["ec"][(u, v)] = d["capacity"]

    data = {None: data}
    # %%

    instance = m.create_instance(data)

    solver = pyo.SolverFactory("gurobi")
    result = solver.solve(instance)

    exchanges = [
        (k[0], k[1], k[2], round(instance.exchange[k](), 2))
        for k in instance.K
        if round(instance.exchange[k](), 2) > 0
    ]

    results = {
        "exchanges": exchanges,
    }
    return results

# %%

if __name__ == "__main__":

    # %%
    import pandas as pd
    import networkx as nx

    # data = {
    #     None: {
    #         "I": ["A", "B", "C"],
    #         "C": [1, 2],
    #         "E": [("A", "B"), ("B", "C")],
    #         "u": {
    #             ("A", 1): 10,
    #             ("A", 2): 10,
    #             ("B", 1): 10,
    #             ("B", 2): 10,
    #             ("C", 1): 10,
    #             ("C", 2): 10,
    #         },
    #         "ec": {("A", "B"): 10, ("B", "C"): 10},
    #     }
    # }

    g_cycle = nx.cycle_graph(6)
    data = {}
    data["I"] = list(g_cycle.nodes())
    data["C"] = list(range(10))
    data["E"] = list(g_cycle.edges())
    # data["E"].extend([(v, u) for (u, v) in data["E"]])
    data["u"] = {}
    for i in data["I"]:
        for c in data["C"]:
            data["u"][(i, c)] = 1000 if (c % 6) == i else 0
    data["ec"] = {}
    for e in data["E"]:
        data["ec"][e] = 1000

    data = {None: data}
    # %%

    instance = m.create_instance(data)

    solver = pyo.SolverFactory("gurobi")
    result = solver.solve(instance)

    # %%
    rows = []
    for i in instance.I:
        chg = []
        for c in instance.C:
            var = sum(
                instance.exchange[j, i, c]()
                for j in instance.I
                if (j, i, c) in instance.exchange
            )
            chg.append(var)
        rows.append(chg)
    df_change = pd.DataFrame(rows).round()
    print(df_change.sum().sum())

    # %%
    edgelist = [
        (k[0], k[1], {"w": f"{k[2]}: {round(instance.exchange[k](), 2)}"})
        for k in instance.K
        if round(instance.exchange[k](), 2) > 0
    ]

    # %%
    for k in instance.K:
        for e in instance.E:
            print(k, e, instance.flows[e, k]())

    # %%
    fig, ax = draw_exchange_graph(edgelist)

    # %%
    import pyvis as pv

    net = pv.network.Network(directed=True)
    for i in data[None]["I"]:
        net.add_node(i, label=f"{i}")
    for u, v, params in edgelist:
        net.add_edge(u, v, title=params["w"])

    net.show_buttons(filter_=["physics"])
    net.set_edge_smooth("dynamic")
    # net.toggle_physics(False)
    net.show("test.html")

# %%
