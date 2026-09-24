# Copyright (c) 2026 Adnane Yacheur. All rights reserved. Contact: adnaneyacheur@gmail.com
"""ASCII FBX 7.4 writer tuned for the Unreal Engine 4 / CARLA importer.

Axis system is written as Z-up, front -Y, right-handed, which is exactly the
system UE4 converts to, so the importer leaves the geometry untouched and
only mirrors Y (x, y, z) -> (x, -y, z). CARLA applies the same mirror to
OpenDRIVE coordinates, so mesh and .xodr line up. Units are centimetres.
"""
import os
import time

import numpy as np

from .. import progress as prog
from ..meshlib import face_normals, vertex_normals

SCALE = 100.0  # metres -> centimetres


def _fmt_floats(values, decimals):
    """Fast float formatting through numpy (returns list of strings)."""
    return np.char.mod(f"%.{decimals}f", np.asarray(values, np.float64).ravel())


def _arr_fast(f, name, values, decimals, indent):
    values = np.asarray(values, np.float64).ravel()
    f.write(f"{indent}{name}: *{len(values)} {{\n{indent}\ta: ")
    per = 3000
    n = len(values)
    for a in range(0, n, per):
        f.write(",".join(_fmt_floats(values[a:a + per], decimals)))
        if a + per < n:
            f.write(",\n")
    f.write(f"\n{indent}}}\n")


def _arr_int(f, name, values, indent):
    values = np.asarray(values, np.int64).ravel()
    f.write(f"{indent}{name}: *{len(values)} {{\n{indent}\ta: ")
    per = 5000
    n = len(values)
    for a in range(0, n, per):
        f.write(",".join(map(str, values[a:a + per].tolist())))
        if a + per < n:
            f.write(",\n")
    f.write(f"\n{indent}}}\n")


def write_fbx(path, objects, materials, tex_base_dir, log=print):
    """objects: Scene.finalize() output. materials: name -> {color, texture}."""
    uid = [1_000_000]

    def new_id():
        uid[0] += 1
        return uid[0]

    mat_ids = {m: new_id() for m in materials}
    tex_ids = {m: (new_id(), new_id()) for m, v in materials.items() if v.get("texture")}
    n_models = len(objects)
    now = time.localtime()
    abs_dir = os.path.dirname(os.path.abspath(path))

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("; FBX 7.4.0 project file\n; Created by duplicat_env\n\n")
        f.write("FBXHeaderExtension:  {\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n")
        f.write("\tCreationTimeStamp:  {\n\t\tVersion: 1000\n")
        f.write(f"\t\tYear: {now.tm_year}\n\t\tMonth: {now.tm_mon}\n\t\tDay: {now.tm_mday}\n")
        f.write(f"\t\tHour: {now.tm_hour}\n\t\tMinute: {now.tm_min}\n\t\tSecond: {now.tm_sec}\n\t\tMillisecond: 0\n\t}}\n")
        f.write('\tCreator: "duplicat_env"\n}\n')
        f.write("GlobalSettings:  {\n\tVersion: 1000\n\tProperties70:  {\n")
        for k, v in (("UpAxis", 2), ("UpAxisSign", 1), ("FrontAxis", 1), ("FrontAxisSign", -1),
                     ("CoordAxis", 0), ("CoordAxisSign", 1), ("OriginalUpAxis", 2), ("OriginalUpAxisSign", 1)):
            f.write(f'\t\tP: "{k}", "int", "Integer", "",{v}\n')
        f.write('\t\tP: "UnitScaleFactor", "double", "Number", "",1\n')
        f.write('\t\tP: "OriginalUnitScaleFactor", "double", "Number", "",1\n')
        f.write('\t\tP: "AmbientColor", "ColorRGB", "Color", "",0,0,0\n')
        f.write('\t\tP: "DefaultCamera", "KString", "", "", "Producer Perspective"\n')
        f.write('\t\tP: "TimeMode", "enum", "", "",11\n')
        f.write("\t}\n}\n\n")

        f.write("Documents:  {\n\tCount: 1\n\tDocument: 900000, \"Scene\", \"Scene\" {\n")
        f.write('\t\tProperties70:  {\n\t\t\tP: "SourceObject", "object", "", ""\n')
        f.write('\t\t\tP: "ActiveAnimStackName", "KString", "", "", ""\n\t\t}\n\t\tRootNode: 0\n\t}\n}\n\n')
        f.write("References:  {\n}\n\n")

        f.write("Definitions:  {\n\tVersion: 100\n")
        total = 1 + 2 * n_models + len(materials) + 2 * len(tex_ids)
        f.write(f"\tCount: {total}\n")
        f.write('\tObjectType: "GlobalSettings" {\n\t\tCount: 1\n\t}\n')
        f.write(f'\tObjectType: "Model" {{\n\t\tCount: {n_models}\n\t}}\n')
        f.write(f'\tObjectType: "Geometry" {{\n\t\tCount: {n_models}\n\t}}\n')
        f.write(f'\tObjectType: "Material" {{\n\t\tCount: {len(materials)}\n\t}}\n')
        if tex_ids:
            f.write(f'\tObjectType: "Texture" {{\n\t\tCount: {len(tex_ids)}\n\t}}\n')
            f.write(f'\tObjectType: "Video" {{\n\t\tCount: {len(tex_ids)}\n\t}}\n')
        f.write("}\n\n")

        f.write("Objects:  {\n")
        connections = []
        for k, obj in enumerate(objects):
            if k % 10 == 0:
                prog.update("fbx", done=k, total=len(objects), detail=f"mesh {k}/{len(objects)}")
            gid, mid = new_id(), new_id()
            name = obj["name"]
            Vs, Fs, UVs, Ns, Ms = [], [], [], [], []
            used_mats = []
            off = 0
            for mat, V, F, UV in obj["parts"]:
                Vs.append(V)
                Fs.append(F + off)
                UVs.append(UV)
                if obj["smooth"]:
                    Ns.append(vertex_normals(V, F)[F].reshape(-1, 3))
                else:
                    Ns.append(np.repeat(face_normals(V, F), 3, axis=0))
                Ms.append(np.full(len(F), len(used_mats)))
                used_mats.append(mat)
                off += len(V)
            V = np.concatenate(Vs) * SCALE
            F = np.concatenate(Fs)
            UV = np.concatenate(UVs)
            N = np.concatenate(Ns)
            M = np.concatenate(Ms)
            pvi = F.copy()
            pvi[:, 2] = -pvi[:, 2] - 1

            f.write(f'\tGeometry: {gid}, "Geometry::{name}", "Mesh" {{\n')
            _arr_fast(f, "Vertices", V, 2, "\t\t")
            _arr_int(f, "PolygonVertexIndex", pvi, "\t\t")
            f.write("\t\tGeometryVersion: 124\n")
            f.write("\t\tLayerElementNormal: 0 {\n\t\t\tVersion: 102\n\t\t\tName: \"\"\n")
            f.write('\t\t\tMappingInformationType: "ByPolygonVertex"\n\t\t\tReferenceInformationType: "Direct"\n')
            _arr_fast(f, "Normals", N, 4, "\t\t\t")
            f.write("\t\t}\n")
            f.write("\t\tLayerElementUV: 0 {\n\t\t\tVersion: 101\n\t\t\tName: \"UVMap\"\n")
            f.write('\t\t\tMappingInformationType: "ByPolygonVertex"\n\t\t\tReferenceInformationType: "IndexToDirect"\n')
            _arr_fast(f, "UV", UV, 4, "\t\t\t")
            _arr_int(f, "UVIndex", F, "\t\t\t")
            f.write("\t\t}\n")
            f.write("\t\tLayerElementMaterial: 0 {\n\t\t\tVersion: 101\n\t\t\tName: \"\"\n")
            if len(used_mats) == 1:
                f.write('\t\t\tMappingInformationType: "AllSame"\n\t\t\tReferenceInformationType: "IndexToDirect"\n')
                _arr_int(f, "Materials", [0], "\t\t\t")
            else:
                f.write('\t\t\tMappingInformationType: "ByPolygon"\n\t\t\tReferenceInformationType: "IndexToDirect"\n')
                _arr_int(f, "Materials", M, "\t\t\t")
            f.write("\t\t}\n")
            f.write("\t\tLayer: 0 {\n\t\t\tVersion: 100\n")
            for le in ("LayerElementNormal", "LayerElementMaterial", "LayerElementUV"):
                f.write(f'\t\t\tLayerElement:  {{\n\t\t\t\tType: "{le}"\n\t\t\t\tTypedIndex: 0\n\t\t\t}}\n')
            f.write("\t\t}\n\t}\n")

            f.write(f'\tModel: {mid}, "Model::{name}", "Mesh" {{\n\t\tVersion: 232\n\t\tProperties70:  {{\n')
            f.write('\t\t\tP: "RotationActive", "bool", "", "",1\n')
            f.write('\t\t\tP: "InheritType", "enum", "", "",1\n')
            f.write('\t\t\tP: "ScalingMax", "Vector3D", "Vector", "",0,0,0\n')
            f.write('\t\t\tP: "DefaultAttributeIndex", "int", "Integer", "",0\n')
            f.write('\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0,0\n')
            f.write('\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0\n')
            f.write('\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1\n')
            f.write("\t\t}\n\t\tShading: T\n\t\tCulling: \"CullingOff\"\n\t}\n")

            connections.append(f'\t;Model::{name}, Model::RootNode\n\tC: "OO",{mid},0\n')
            connections.append(f'\t;Geometry::{name}, Model::{name}\n\tC: "OO",{gid},{mid}\n')
            for mat in used_mats:
                connections.append(f'\t;Material::{mat}, Model::{name}\n\tC: "OO",{mat_ids[mat]},{mid}\n')
            if (k + 1) % 50 == 0:
                log(f"  FBX: {k + 1}/{n_models} meshes")

        for mat, info in materials.items():
            r, g, b = info["color"]
            f.write(f'\tMaterial: {mat_ids[mat]}, "Material::{mat}", "" {{\n\t\tVersion: 102\n')
            f.write('\t\tShadingModel: "lambert"\n\t\tMultiLayer: 0\n\t\tProperties70:  {\n')
            f.write('\t\t\tP: "AmbientColor", "Color", "", "A",0,0,0\n')
            f.write(f'\t\t\tP: "DiffuseColor", "Color", "", "A",{r:.4f},{g:.4f},{b:.4f}\n')
            f.write('\t\t\tP: "DiffuseFactor", "Number", "", "A",1\n')
            f.write('\t\t\tP: "Emissive", "Vector3D", "Vector", "",0,0,0\n')
            f.write('\t\t\tP: "Ambient", "Vector3D", "Vector", "",0,0,0\n')
            f.write(f'\t\t\tP: "Diffuse", "Vector3D", "Vector", "",{r:.4f},{g:.4f},{b:.4f}\n')
            f.write('\t\t\tP: "Opacity", "double", "Number", "",1\n')
            f.write("\t\t}\n\t}\n")
            if mat in tex_ids:
                tid, vid = tex_ids[mat]
                rel = info["texture"].replace("/", os.sep)
                absf = os.path.join(abs_dir, rel)
                tname = os.path.splitext(os.path.basename(rel))[0]
                f.write(f'\tVideo: {vid}, "Video::{tname}", "Clip" {{\n\t\tType: "Clip"\n')
                f.write(f'\t\tProperties70:  {{\n\t\t\tP: "Path", "KString", "XRefUrl", "", "{absf}"\n\t\t}}\n')
                f.write(f'\t\tUseMipMap: 0\n\t\tFilename: "{absf}"\n\t\tRelativeFilename: "{rel}"\n\t}}\n')
                f.write(f'\tTexture: {tid}, "Texture::{tname}", "" {{\n\t\tType: "TextureVideoClip"\n\t\tVersion: 202\n')
                f.write(f'\t\tTextureName: "Texture::{tname}"\n\t\tProperties70:  {{\n')
                f.write('\t\t\tP: "UVSet", "KString", "", "", "UVMap"\n\t\t\tP: "UseMaterial", "bool", "", "",1\n\t\t}\n')
                f.write(f'\t\tMedia: "Video::{tname}"\n\t\tFileName: "{absf}"\n\t\tRelativeFilename: "{rel}"\n')
                f.write("\t\tModelUVTranslation: 0,0\n\t\tModelUVScaling: 1,1\n\t\tTexture_Alpha_Source: \"None\"\n")
                f.write("\t\tCropping: 0,0,0,0\n\t}\n")
                connections.append(f'\t;Texture::{tname}, Material::{mat}\n\tC: "OP",{tid},{mat_ids[mat]}, "DiffuseColor"\n')
                connections.append(f'\t;Video::{tname}, Texture::{tname}\n\tC: "OO",{vid},{tid}\n')
        f.write("}\n\n")

        f.write("Connections:  {\n")
        f.writelines(connections)
        f.write("}\n")
    return path
