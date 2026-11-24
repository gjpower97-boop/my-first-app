"""Write a Bambu Studio style 3MF: one object made of several parts,
each part pre-assigned to a filament slot (so colours can be changed per part)."""
import uuid
import zipfile
import numpy as np


def _mesh_xml(oid, verts, faces):
    v = "\n".join(f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in verts)
    t = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces)
    return (f'  <object id="{oid}" p:UUID="{uuid.uuid4()}" type="model">\n   <mesh>\n    <vertices>\n{v}\n'
            f'    </vertices>\n    <triangles>\n{t}\n    </triangles>\n   </mesh>\n  </object>\n')


NS = ('xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
      'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
      'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"')
IDENT = "1 0 0 0 1 0 0 0 1 0 0 0"


def write_bambu_3mf(path, parts, object_name, bed_xy=(100, 100)):
    """parts: list of (part_name, trimesh.Trimesh, filament_slot starting at 1)."""
    # recentre the whole object on x/y and put it on z=0
    allv = np.vstack([m.vertices for _, m, _ in parts])
    off = np.array([-(allv[:, 0].min() + allv[:, 0].max()) / 2,
                    -(allv[:, 1].min() + allv[:, 1].max()) / 2, -allv[:, 2].min()])
    parent = len(parts) + 1

    objs = "".join(_mesh_xml(i + 1, m.vertices + off, m.faces) for i, (_, m, _) in enumerate(parts))
    sub = f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" {NS}>\n' \
          f' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n{objs} </resources>\n <build/>\n</model>\n'

    comps = "".join(f'    <component p:path="/3D/Objects/object_1.model" objectid="{i + 1}" '
                    f'p:UUID="{uuid.uuid4()}" transform="{IDENT}"/>\n' for i in range(len(parts)))
    main = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" {NS}>\n'
            f' <metadata name="Application">BambuStudio-01.10.00.00</metadata>\n'
            f' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
            f' <metadata name="Title">{object_name}</metadata>\n'
            f' <resources>\n  <object id="{parent}" p:UUID="{uuid.uuid4()}" type="model">\n   <components>\n{comps}'
            f'   </components>\n  </object>\n </resources>\n <build p:UUID="{uuid.uuid4()}">\n'
            f'  <item objectid="{parent}" p:UUID="{uuid.uuid4()}" transform="1 0 0 0 1 0 0 0 1 {bed_xy[0]} {bed_xy[1]} 0" printable="1"/>\n'
            f' </build>\n</model>\n')

    part_cfg = "".join(
        f'    <part id="{i + 1}" subtype="normal_part">\n'
        f'      <metadata key="name" value="{name}"/>\n'
        f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
        f'      <metadata key="extruder" value="{slot}"/>\n'
        f'      <mesh_stat edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
        f'    </part>\n' for i, (name, _, slot) in enumerate(parts))
    cfg = (f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <object id="{parent}">\n'
           f'    <metadata key="name" value="{object_name}"/>\n    <metadata key="extruder" value="{parts[0][2]}"/>\n'
           f'{part_cfg}  </object>\n  <plate>\n    <metadata key="plater_id" value="1"/>\n'
           f'    <metadata key="plater_name" value=""/>\n    <metadata key="locked" value="false"/>\n'
           f'    <model_instance>\n      <metadata key="object_id" value="{parent}"/>\n'
           f'      <metadata key="instance_id" value="0"/>\n      <metadata key="identify_id" value="{parent * 100}"/>\n'
           f'    </model_instance>\n  </plate>\n  <assemble>\n'
           f'   <assemble_item object_id="{parent}" instance_id="0" transform="1 0 0 0 1 0 0 0 1 {bed_xy[0]} {bed_xy[1]} 0" offset="0 0 0" />\n'
           f'  </assemble>\n</config>\n')

    ctypes = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
              ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
              ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
              ' <Default Extension="png" ContentType="image/png"/>\n'
              ' <Default Extension="config" ContentType="text/xml"/>\n</Types>\n')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            '</Relationships>\n')
    mrels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
             ' <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
             '</Relationships>\n')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ctypes)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", main)
        z.writestr("3D/_rels/3dmodel.model.rels", mrels)
        z.writestr("3D/Objects/object_1.model", sub)
        z.writestr("Metadata/model_settings.config", cfg)
