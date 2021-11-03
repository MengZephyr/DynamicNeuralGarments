import bpy
import bmesh
import os
import random
from math import *
from mathutils import *
import mathutils

C = bpy.context
D = bpy.data

sk = D.scenes.keys()[0]
myCamera = D.scenes[sk].camera
myViewer = C.view_layer
pmm = myCamera.calc_matrix_camera(C.evaluated_depsgraph_get())
sce = C.scene
Arma = D.objects['Armature']


def randFloat(a, b):
    return random.random()*(b-a) + a


def readOBJFile(fname):
    if not(os.path.exists(fname)):
        return None, None
    vertArray = []
    faceArray = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            vertArray.append([v[0], v[1], v[2]])
        if values[0] == 'f':
            f = [int(x.split('/')[0])-1 for x in values[1:4]]
            faceArray.append(f)
    file.close()
    return vertArray, faceArray


def writeMatrix(wmm, pmm, cpos, fileName):
    with open(fileName, 'w') as f:
        for c in range(len(wmm)):
            f.write(str(wmm[c][0]) + " " + str(wmm[c][1]) + " " + str(wmm[c][2]) + " " + str(wmm[c][3]) + "\n")
        f.write("\n")
        for c in range(len(pmm)):
            f.write(str(pmm[c][0]) + " " + str(pmm[c][1]) + " " + str(pmm[c][2]) + " " + str(pmm[c][3]) + "\n")
        f.write("\n")
        f.write(str(cpos[0]) + " " + str(cpos[1]) + " " + str(cpos[2]) + " " + "1." + "\n")
        f.write("\n")
        f.close()

        
def writeList(dlist, fileName):
    with open(fileName, 'w') as f:
        f.write(str(len(dlist)) + "\n")
        for d in dlist:
            f.write(str(d) + " ")
        f.write("\n")
        f.close()

        
def readList(fileName):
    if not(os.path.exists(fileName)):
        return None
    
    with open(fileName, 'r') as f:
        l = f.readline()
        d = l.split()
        dLen = int(d[0])
        
        l = f.readline()
        d = l.split()
        dlist = [float(x) for x in d[0:dLen]]
        f.close()
    return dlist

#-----------------------------------------------------------------

#pre_FileName = 'D:/models/MD/DataModel/DressOri/case_1/ShortDress/chessboard/PD10_0000001.obj'
#bpy.ops.import_scene.obj(filepath=pre_FileName, split_mode='OFF')

#premyobj = C.selected_objects[0]
#premyobj.name = 'pre_Dress_obj'
#premymeshes = premyobj.data
#premymeshes.name = 'pre_Dress_mesh'
#mat = D.materials.get('Dress_mat')
#premyobj.active_material = mat

#-----------------------------------------------------------------

#FileName = 'D:/models/MD/DataModel/DressOri/case_1/ShortDress/chessboard/PD10_0000002.obj'
#bpy.ops.import_scene.obj(filepath=FileName, split_mode='OFF')

#myobj = C.selected_objects[0]
#myobj.name = 'Dress_obj'
#mymeshes = myobj.data
#mymeshes.name = 'Dress_mesh'
#mat = D.materials.get('Dress_mat')
#myobj.active_material = mat

#-----------------------------------------------------------------

#FileName = 'D:/models/MD/DataModel/DressOri/case_1/ShortDress/chessboard/PD10_0000002.obj'
#bpy.ops.import_scene.obj(filepath=FileName, split_mode='OFF')

#next_myobj = C.selected_objects[0]
#next_myobj.name = 'next_Dress_obj'
#nextmymeshes = next_myobj.data
#nextmymeshes.name = 'next_Dress_mesh'
#mat = D.materials.get('Dress_mat')
#next_myobj.active_material = mat

#-----------------------------------------------------------------

pre_obj = D.objects['pre_Dress_obj']
pre_meshes = pre_obj.data

cur_obj = D.objects['Dress_obj']
cur_meshes = cur_obj.data

next_obj = D.objects['next_Dress_obj']
next_meshes = next_obj.data

mat_name = "mater_Chama"
mat = D.materials.get(mat_name)
if mat is None:
    mat = D.materials.new(mat_name)
#mat.diffuse_color = (0.185, 0.259, 1., 1.)
mat.diffuse_color = (0.507, 0.196, 1., 1.)
#colors = [(34./255., 32./255., 188./255., 1.), (45./255., 143./255., 69./255., 1), 
#(150./255., 141./255., 30./255., 1.), (157./255., 70./255., 193./255., 1.), 
#(135./255., 41./255., 26./255., 1.)]

minFrame = 1
maxFrame = 849
iniRz = -25

Frame0 = 2
Frame1 = 848
FrameID = [i for i in range(Frame0, Frame1+1)]
ifGarment = True
ifSaveCamera = False

numView = 10
delta = 1./float(numView)

ObjRoot = 'D:/models/MD/DataModel/DressOri/case_1/Chamuse_Complex/10_L/'
saveRoot = 'D:/models/NR/Data/ver_6/case_1/multiLaces/v_train/'
offListRoot = 'D:/models/NR/Data/ver_6/case_1/cList/'


CameraBox = D.objects['Empty']

sce.frame_set(max(Frame0-1, minFrame))
if ifGarment is True:
    preObjFile = ObjRoot + 'PD10_' + str(max(Frame0, minFrame+1)).zfill(7) + '.obj'
    vertArra, faceArray = readOBJFile(preObjFile)
    C.view_layer.objects.active = pre_obj
    bpy.ops.object.mode_set(mode='EDIT')
    prebm = bmesh.new()
    prebm.from_mesh(pre_meshes)
    i = 0
    for v in prebm.verts:
        v.co = Vector(vertArra[i])
        i += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    prebm.to_mesh(pre_meshes)
    pre_meshes.update()
    prebm.free()
    
    ObjFile = ObjRoot + 'PD10_' + str(Frame0+1).zfill(7) + '.obj'
    vertArra, faceArray = readOBJFile(ObjFile)
    C.view_layer.objects.active = cur_obj
    bpy.ops.object.mode_set(mode='EDIT')
    cur_bm = bmesh.new()
    cur_bm.from_mesh(cur_meshes)
    i = 0
    for v in cur_bm.verts:
        v.co = Vector(vertArra[i])
        i += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    cur_bm.to_mesh(cur_meshes)
    cur_meshes.update()
    cur_bm.free()
    

colourIDCC= 0
for fID in FrameID:
    if fID >= maxFrame:
        break
    
    if ifGarment is True:
        ObjFile = ObjRoot + 'PD10_' + str(fID+2).zfill(7) + '.obj'
        vertArra, faceArray = readOBJFile(ObjFile)
        C.view_layer.objects.active = next_obj
        bpy.ops.object.mode_set(mode='EDIT')
        next_bm = bmesh.new()
        next_bm.from_mesh(next_meshes)
        i = 0
        for v in next_bm.verts:
            v.co = Vector(vertArra[i])
            i += 1
        bpy.ops.object.mode_set(mode='OBJECT')
        next_bm.to_mesh(next_meshes)
        next_meshes.update()
        next_bm.free()
        
    offSetList = readList(offListRoot + str(fID+1).zfill(7) + '.txt')
    numView = len(offSetList)
    
#    offSetList = []
#    begA = min(randFloat(0., 1.), 1)
    
    for vID in range(numView):
        C.view_layer.objects.active = CameraBox
        
        begA = offSetList[vID]
        
#        begA = begA + delta
#        if begA >= 1.:
#            begA = begA - 1.
            
        C.object.constraints["Follow Path"].offset_factor = begA
        bpy.data.worlds["World"].node_tree.nodes.get("Mapping").rotation[2] = radians(C.object.constraints["Follow Path"].offset_factor*360 + iniRz)  
        #offSetList.append(begA)
#        gColor = colors[ colourIDCC % len(colors)]
#        colourIDCC += 1
#        mat.diffuse_color = gColor
        
        sce.frame_set(max(fID-1, minFrame))
        next_obj.hide_render=True
        cur_obj.hide_render=True
        pre_obj.hide_render = True
        D.scenes[sk].render.filepath = saveRoot + 'bg/p_' + str(fID+1).zfill(7) + '_' + str(vID) + '.png'
        bpy.ops.render.render(write_still=True)
        
        next_obj.hide_render=True
        cur_obj.hide_render=True
        pre_obj.hide_render = False
        D.scenes[sk].render.filepath = saveRoot + 'render_1/p_' + str(fID+1).zfill(7) + '_' + str(vID) + '.png'
        bpy.ops.render.render(write_still=True)
        
        sce.frame_set(fID)
        next_obj.hide_render=True
        cur_obj.hide_render=True
        pre_obj.hide_render = True
        D.scenes[sk].render.filepath = saveRoot + 'bg/' + str(fID+1).zfill(7) + '_' + str(vID) + '.png'
        bpy.ops.render.render(write_still=True)
        
        next_obj.hide_render=True    
        cur_obj.hide_render=False
        pre_obj.hide_render = True
        D.scenes[sk].render.filepath = saveRoot + 'render_1/' + str(fID+1).zfill(7) + '_' + str(vID) + '.png'
        bpy.ops.render.render(write_still=True)
        
        sce.frame_set(fID+1)
        next_obj.hide_render=False    
        cur_obj.hide_render=True
        pre_obj.hide_render = True
        D.scenes[sk].render.filepath = saveRoot + 'render_1/a_' + str(fID+1).zfill(7) + '_' + str(vID) + '.png'
        bpy.ops.render.render(write_still=True)
        
        if ifSaveCamera is True:
            wmm = myCamera.matrix_world.inverted()
            npos = myCamera.matrix_world.translation
            writeMatrix(wmm, pmm, npos, saveRoot + 'cameras/' + str(fID+1).zfill(7) + '_' + str(vID) + '_c.txt')
        
    C.view_layer.objects.active = cur_obj
    bpy.ops.object.mode_set(mode='EDIT')
    cur_bm = bmesh.new()
    cur_bm.from_mesh(cur_meshes)
    bpy.ops.object.mode_set(mode='OBJECT')
    cur_bm.to_mesh(pre_meshes)
    pre_meshes.update()
    cur_bm.free()
    
    C.view_layer.objects.active = next_obj
    bpy.ops.object.mode_set(mode='EDIT')
    next_bm = bmesh.new()
    next_bm.from_mesh(next_meshes)
    bpy.ops.object.mode_set(mode='OBJECT')
    next_bm.to_mesh(cur_meshes)
    cur_meshes.update()
    next_bm.free()
    
    #writeList(offSetList, saveRoot + 'cList/' + str(fID+1).zfill(7) + '.txt')

