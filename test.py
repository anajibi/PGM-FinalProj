from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian

test = sample_erdos_renyi_linear_gaussian(
    num_variables=10,
    num_edges=25,
)

print(test)