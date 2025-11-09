#!/usr/bin/env python3
"""
Created on 2025-11-06 (Thu) 13:46:31

feature extraction utils

@author: I.Azuma
"""
# %%
# JSONを読み込む
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

records = []
for feature in data["features"]:
    props = feature["properties"]
    geom = feature["geometry"]

    label = props.get("classification", {}).get("name", None)
    color = props.get("classification", {}).get("color", None)
    geom_type = geom["type"]
    coords = geom["coordinates"]

    # shapelyで扱いやすい形に変換
    polygon = shape(geom)
    minx, miny, maxx, maxy = polygon.bounds
    area = polygon.area

    records.append({
        "id": feature.get("id"),
        "label": label,
        "color_rgb": color,
        "geometry_type": geom_type,
        "n_points": len(coords[0]) if geom_type == "Polygon" else None,
        "bbox_xmin": minx,
        "bbox_ymin": miny,
        "bbox_xmax": maxx,
        "bbox_ymax": maxy,
        "area": area,
        "coordinates": coords[0] if geom_type == "Polygon" else coords,
    })
df = pd.DataFrame(records)

import h5py
import openslide
import matplotlib.pyplot as plt

wsi_path = f"/workspace/HDDX/Pathology_datasource/BRACS/BRACS_WSI/test/Group_MT/Type_IC/{file_name}.svs"
h5_path = f"{BASE_DIR}/datasource/BRACS/CLAM_v2/Group_MT/Type_IC/features/feats_h5/{file_name}.h5"

wsi = openslide.open_slide(wsi_path)

hdf5_file = h5py.File(h5_path, "r")
coords = hdf5_file['coords'][:]

for idx, row in df.iterrows():
    bbox = (row['bbox_xmin'], row['bbox_ymin'], row['bbox_xmax'], row['bbox_ymax'])
    print(f"Annotation ID: {row['id']}, Label: {row['label']}, BBox: {bbox}, Area: {row['area']}")
    
    # 座標リストを表示
    print(f"Coordinates: {row['coordinates'][:5]}...")  # 最初の5点だけ表示

    # WSI上での位置を確認
    for coord in coords:
        x, y = coord
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            print(f"  - Patch at ({x}, {y}) is within the annotation bbox.")
    
    img = wsi.read_region((int(bbox[0]), int(bbox[1])), 0, (int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1]))).convert('RGB')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
