import pathlib

manifest = {
  "videos": [
    {"id": str(id), "path": f"videos/{id}.mp4"}
      for id in pathlib.Path('./DanceProj1/videos').glob('*.mp4')
  ],
  "genres": [{"name": "Ballet Jazz", "id": "JB"}, {"name": "Break", "id": "BR"}, {"name": "House", "id": "HO"}, {"name": "Krump", "id": "KR"}, {"name": "LA style Hip-hop", "id": "LH"}, {"name": "Lock", "id": "LO"}, {"name": "Middle Hip-hop", "id": "MH"}, {"name": "Pop", "id": "PO"}, {"name": "Street Jazz", "id": "JS"}, {"name": "Waack", "id": "WA"}]
}

import json
with open("manifest.json", 'w') as fh:
  json.dump(manifest, fh)