# GaitNet


NEU_CAI = {'data': ndarray(1272, 84, 18), 'labels': ndarray(1272,), 'genders': ndarray(1272,), 'subject_labels': ndarray(32,), 'subject_genders': ndarray(32,),
                   'cycleIdx': ndarray(32, 2), 'num_subjects': int, 'num_cycles': int, 'node_types': list(1272),
                   'edge_indexs': list(1272), 'edge_features': list(1272), 'edge_types': list(1272)}

data: ndarray(1272, 84, 18) is 18-channel gait cycle data, i.e., node features.

labels: ndarray(1272,) is gait cycle labels, 1: CAI, 0: health.

genders: ndarray(1272,) is gait cycle genders, 1: female, 0: male.

subject_labels: ndarray(32,) is subject labels, 1: CAI, 0: health.

subject_genders: ndarray(32,) is subject genders, 1: female, 0: male.

cycleIdx: ndarray(32, 2) is subject start and end gait cycle index.

num_subjects: int is the number of subjects.

num_cycles: int is the number of cycles.

node_types: list(1272) is node types, i.e. nodes of X,Y,Z type.

edge_indexs: list(1272) is edge indexs by creating kNN-Graph (Graph connectivity in COO format with shape [2, num_edges]).

edge_features: list(1272) is edge features.

edge_types: list(1272) is edge types, 1: inner edge, 0: outer edge.
