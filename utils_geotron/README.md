Modification of the loc-mode-pred model and pipeline to work with GeoTron data and features.

Config flags:
  NEW-
  use_clusters: True
  predict_clusters: True
  predict_intra_cluster: True
  total_cluster_num: 1060
  max_intra_cluster_num: 1000
  # Maybe some flags need to be added for loss functions for cluster and intra-cluster id
  embed_time_to_next: True
  
  if_embed_loc: True
  if_embed_user: True

  if_embed_time: True

  removed mode embedding