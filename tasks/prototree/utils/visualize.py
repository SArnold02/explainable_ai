import os
import numpy as np
import copy
import argparse
from subprocess import check_call
from PIL import Image
import torch
import math
from tasks.prototree.model import ProtoTree, Node

# Taken from the original repo
def gen_vis(
        tree: ProtoTree,
        arguments: argparse.Namespace,
        classes: tuple
    ) -> None:
    # Helper function to visualize the prediction of the tree
    destination_folder = os.path.join(arguments.output_path, "tree_vis")
    upsample_dir = os.path.join(
        os.path.join(arguments.output_path, "images"),
        "upsample"
    )
    os.makedirs(destination_folder + '/node_vis', exist_ok=True)

    # Create the visualization for the nodes
    with torch.no_grad():
        s = 'digraph T {margin=0;ranksep=".03";nodesep="0.05";splines="false";\n'
        s += 'node [shape=rect, label=""];\n'
        s += _gen_dot_nodes(tree._root, destination_folder, upsample_dir, classes)
        s += _gen_dot_edges(tree._root, classes)[0]
        s += '}\n'

    # Save the visualization
    with open(os.path.join(destination_folder, 'treevis.dot'), 'w') as f:
        f.write(s)
   
    from_p = os.path.join(destination_folder,'treevis.dot')
    to_pdf = os.path.join(destination_folder,'treevis.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s'%(from_p, to_pdf), shell=True)

def _node_vis(node: Node, upsample_dir: str):
    # Chose the right visualization
    if node.is_leaf_node:
        return _leaf_vis(node)
    else:
        return _branch_vis(node, upsample_dir)

def _leaf_vis(node: Node):
    # Function taken from the original implementation
    if node._log_probabilities:
        ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
    else:
        ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
    
    ws = np.ones(ws.shape) - ws
    ws *= 255
    height = 24

    if ws.shape[0] < 36:
        img_size = 36
    else:
        img_size = ws.shape[0]
    scaler = math.ceil(img_size/ws.shape[0])

    img = Image.new('F', (ws.shape[0]*scaler, height))
    pixels = img.load()

    for i in range(scaler*ws.shape[0]):
        for j in range(height-10):
            pixels[i,j]=ws[int(i/scaler)]
        for j in range(height-10,height-9):
            pixels[i,j]=0
        for j in range(height-9,height):
            pixels[i,j]=255

    if scaler*ws.shape[0]>100:
        img=img.resize((100,height))
    return img


def _branch_vis(node: Node, upsample_dir: str):
    # Function taken from original implementation
    branch_id = node.index()
    
    img = Image.open(os.path.join(upsample_dir, '%s_nearest_patch_of_image.png'%branch_id))
    bb = Image.open(os.path.join(upsample_dir, '%s_bounding_box_nearest_patch_of_image.png'%branch_id))
    w, h = img.size
    wbb, hbb = bb.size
    
    if wbb < 100 and hbb < 100:
        cs = wbb, hbb
    else:
        cs = 100/wbb, 100/hbb
        min_cs = min(cs)
        bb = bb.resize(size=(int(min_cs * wbb), int(min_cs * hbb)))
        wbb, hbb = bb.size

    if w < 100 and h < 100:
        cs = w, h
    else:
        cs = 100/w, 100/h
        min_cs = min(cs)
        img = img.resize(size=(int(min_cs * w), int(min_cs * h)))
        w, h = img.size

    between = 4
    total_w = w+wbb + between
    total_h = max(h, hbb)

    together = Image.new(img.mode, (total_w, total_h), color=(255,255,255))
    together.paste(img, (0, 0))
    together.paste(bb, (w+between, 0))

    return together

def _gen_dot_nodes(node: Node, destination_folder: str, upsample_dir: str, classes:tuple):
    # Create the images for the nodes 
    img = _node_vis(node, upsample_dir).convert('RGB')
    
    if node.is_leaf_node:
        # Get the prediction
        if node._log_probabilities:
            distribution = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            distribution = copy.deepcopy(node.distribution().cpu().detach().numpy())
        argmax = np.argmax(distribution)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        
        # Get the class names
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        str_targets = ','.join(t for t in class_targets) if len(class_targets) > 0 else ""
        str_targets = str_targets.replace('_', ' ')
    
    # Create the file
    filename = '{}/node_vis/node_{}_vis.jpg'.format(destination_folder, node.index())
    img.save(filename)
    
    # Chose the right method 
    if node.is_leaf_node:
        s = '{}[imagepos="tc" imagescale=height image="{}" label="{}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'\
        .format(node.index(), filename, str_targets)
    else:
        s = '{}[image="{}" xlabel="{}" fontsize=6 labelfontcolor=gray50 fontname=Helvetica];\n'\
            .format(node.index(), filename, node.index())
    if not node.is_leaf_node:
        return s\
               + _gen_dot_nodes(node.left_node, destination_folder, upsample_dir, classes)\
               + _gen_dot_nodes(node.right_node, destination_folder, upsample_dir, classes)
    return s

def _gen_dot_edges(node: Node, classes:tuple):
    # Create the command for the edges
    if not node.is_leaf_node:
        edge_l, targets_l = _gen_dot_edges(node.left_node, classes)
        edge_r, targets_r = _gen_dot_edges(node.right_node, classes)
        s = '{} -> {} [label="Absent" fontsize=10 tailport="s" headport="n" fontname=Helvetica];"\
            "\n {} -> {} [label="Present" fontsize=10 tailport="s" headport="n" fontname=Helvetica];\n'\
                .format(node.index(), node.left_node.index(), node.index(), node.right_node.index())
        return s + edge_l + edge_r, sorted(list(set(targets_l + targets_r)))

    # Creat if for the leaf node
    if node._log_probabilities:
        distribution = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
    else:
        distribution = copy.deepcopy(node.distribution().cpu().detach().numpy())
    argmax = np.argmax(distribution)
    targets = [argmax] if argmax.shape == () else argmax.tolist()
    class_targets = copy.deepcopy(targets)
    for i in range(len(targets)):
        t = targets[i]
        class_targets[i] = classes[t]
    return '', class_targets
