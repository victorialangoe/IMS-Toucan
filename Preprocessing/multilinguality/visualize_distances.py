import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import torch
from tqdm import tqdm

from Modules.ToucanTTS.InferenceToucanTTS import ToucanTTS
from Utility.utils import load_json_from_path

distance_types = ["tree", "asp", "map", "learned", "l1"]
distance_type = distance_types[2]  # switch here
edge_threshold = 0.1

cache_root = "."
supervised_iso_codes = load_json_from_path(os.path.join(cache_root, "supervised_languages.json"))

if distance_type == "l1":
    iso_codes_to_ids = load_json_from_path(os.path.join(cache_root, "iso_lookup.json"))[-1]
    model_path = "../../Models/ToucanTTS_Meta/best.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    embedding_provider = ToucanTTS(weights=checkpoint["model"], config=checkpoint["config"]).encoder.language_embedding
    embedding_provider.requires_grad_(False)
    l1_dist = dict()
    seen_langs = set()
    for lang_1 in supervised_iso_codes:
        if lang_1 not in seen_langs:
            seen_langs.add(lang_1)
            l1_dist[lang_1] = dict()
        for lang_2 in supervised_iso_codes:
            if lang_2 not in seen_langs:  # it's symmetric
                l1_dist[lang_1][lang_2] = torch.nn.functional.mse_loss(embedding_provider(torch.LongTensor([iso_codes_to_ids[lang_1]])).squeeze(), embedding_provider(torch.LongTensor([iso_codes_to_ids[lang_2]])).squeeze())
    largest_value_l1_dist = 0.0
    for _, values in l1_dist.items():
        for _, value in values.items():
            largest_value_l1_dist = max(largest_value_l1_dist, value)
    for key1 in l1_dist:
        for key2 in l1_dist[key1]:
            l1_dist[key1][key2] = l1_dist[key1][key2] / largest_value_l1_dist
    distance_measure = l1_dist

if distance_type == "tree":
    tree_lookup_path = os.path.join(cache_root, "lang_1_to_lang_2_to_tree_dist.json")
    tree_dist = load_json_from_path(tree_lookup_path)
    distance_measure = tree_dist

if distance_type == "map":
    map_lookup_path = os.path.join(cache_root, "lang_1_to_lang_2_to_map_dist.json")
    map_dist = load_json_from_path(map_lookup_path)
    largest_value_map_dist = 0.0
    for _, values in map_dist.items():
        for _, value in values.items():
            largest_value_map_dist = max(largest_value_map_dist, value)
    for key1 in map_dist:
        for key2 in map_dist[key1]:
            map_dist[key1][key2] = map_dist[key1][key2] / largest_value_map_dist
    distance_measure = map_dist

if distance_type == "learned":
    learned_lookup_path = os.path.join(cache_root, "lang_1_to_lang_2_to_map_dist.json")
    learned_dist = load_json_from_path(learned_lookup_path)
    largest_value_learned_dist = 0.0
    for _, values in learned_dist.items():
        for _, value in values.items():
            largest_value_learned_dist = max(largest_value_learned_dist, value)
    for key1 in learned_dist:
        for key2 in learned_dist[key1]:
            learned_dist[key1][key2] = learned_dist[key1][key2] / largest_value_learned_dist
    distance_measure = learned_dist

if distance_type == "asp":
    asp_dict_path = os.path.join(cache_root, "asp_dict.pkl")
    with open(asp_dict_path, 'rb') as dictfile:
        asp_sim = pickle.load(dictfile)
    lang_list = list(asp_sim.keys())
    asp_dist = dict()
    seen_langs = set()
    for lang_1 in lang_list:
        if lang_1 not in seen_langs:
            seen_langs.add(lang_1)
            asp_dist[lang_1] = dict()
        for index, lang_2 in enumerate(lang_list):
            if lang_2 not in seen_langs:  # it's symmetric
                asp_dist[lang_1][lang_2] = 1 - asp_sim[lang_1][index]
    distance_measure = asp_dist

iso_codes_to_names = load_json_from_path(os.path.join(cache_root, "iso_to_fullname.json"))
distances = list()

for lang_1 in distance_measure:
    if lang_1 not in iso_codes_to_names:
        continue
    for lang_2 in distance_measure[lang_1]:
        distances.append((iso_codes_to_names[lang_1], iso_codes_to_names[lang_2], distance_measure[lang_1][lang_2]))

# Create a graph
G = nx.Graph()

# Add edges along with distances as weights
min_dist = min(d for _, _, d in distances)
max_dist = max(d for _, _, d in distances)
normalized_distances = [(entity1, entity2, (d - min_dist) / (max_dist - min_dist)) for entity1, entity2, d in distances]

for entity1, entity2, d in tqdm(normalized_distances):
    if d <= edge_threshold and entity1 != entity2:
        spring_tension = edge_threshold - d
        G.add_edge(entity1, entity2, weight=spring_tension * 10)

# Draw the graph
pos = nx.spring_layout(G, weight="weight")  # Positions for all nodes
edges = G.edges(data=True)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=1, alpha=0.01)

# Draw edges with labels
nx.draw_networkx_edges(G, pos, alpha=0.01, edge_color="gray")
# nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in edges})

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

plt.title(f'Graph of {distance_type} Distances')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout(pad=0)

plt.show()
