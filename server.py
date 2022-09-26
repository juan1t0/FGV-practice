import os
from pydoc import render_doc
import flask
import numpy as np
import argparse
import json
import csv

from flask import Flask
from flask_cors import CORS
import math

# create Flask app
app = Flask(__name__)
CORS(app)

# --- these will be populated in the main --- #

# list of attribute names of size m
# m -> 20
attribute_names=None

# a 2D numpy array containing binary attributes - it is of size n x m, for n paintings and m attributes
# shape -> 403x20
painting_attributes=None

# a list of epsiode names of size n
# n -> 403
episode_names=None

# a list of painting image URLs of size n
# n-> 403
painting_image_urls=None

pca = None
pca_ = None
km = None

@app.route('/get_points_data', methods=['GET'])
def get_points_data():
    return flask.jsonify({'names':episode_names,'urls':painting_image_urls})
#

'''
This will return an array of strings containing the episode names -> these should be displayed upon hovering over circles.
'''
@app.route('/get_episode_names', methods=['GET'])
def get_episode_names():
    return flask.jsonify(episode_names)
#

'''
This will return an array of URLs containing the paths of images for the paintings
'''
@app.route('/get_painting_urls', methods=['GET'])
def get_painting_urls():
    return flask.jsonify(painting_image_urls)
#

'''
TODO: implement PCA, this should return data in the same format as you saw in the first part of the assignment:
    * the 2D projection
    * x loadings, consisting of pairs of attribute name and value
    * y loadings, consisting of pairs of attribute name and value
'''
@app.route('/initial_pca', methods=['GET'])
def initial_pca():
    # data_m = painting_attributes - np.mean(painting_attributes, axis=0)
    # cm = np.cov(data_m, rowvar=False)
    # e_val, e_vec = np.linalg.eigh(cm)
    # index = np.argsort(e_val)[::-1]
    # se_val = e_val[index]
    # se_vec = e_vec[:,index]
    
    # subset = se_vec[:,0:2]
    # pca = np.dot(subset.T, data_m.T).T
    # loads = subset.T * np.sqrt(se_val.reshape(1,len(attribute_names)))
    global pca_
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    painting_attributes_ = scaler.fit_transform(painting_attributes)
    pca_ = pca.fit_transform(painting_attributes_)
    loads = pca.components_.T * np.sqrt(pca.explained_variance_)

    x_loading = [{'attribute':a,'loading':l.item()} for a,l in zip(attribute_names,loads[:,0])]
    y_loading = [{'attribute':a,'loading':l.item()} for a,l in zip(attribute_names,loads[:,1])]

    pca_data = {'loading_x':x_loading, 'loading_y':y_loading, 'projection': pca_.tolist()}
    return flask.jsonify(pca_data)
#

'''
TODO: implement ccPCA here. This should return data in _the same format_ as initial_pca above.
It will take in a list of data items, corresponding to the set of items selected in the visualization. This can be acquired from `flask.request.json`. This should be a list of data item indices - the **target set**.
The alpha value, from the paper, should be set to 1.1 to start, though you are free to adjust this parameter.
'''
@app.route('/ccpca', methods=['GET','POST'])
def ccpca():
    #global km
    if flask.request.method == 'POST':
        data = flask.request.form
        selected_idx = []
        kmlabels = km.labels_
        targetlbls = [int(k) for k in data if data[k]=='true']
        for i,l in enumerate(kmlabels):
            if l.item() in targetlbls:
                selected_idx += [i]
        alpha = 1.1
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        painting_attributes_ = scaler.fit_transform(painting_attributes)
        back_idx = [i for i in range(painting_attributes.shape[0]) if i not in selected_idx]
        target_set = painting_attributes_[selected_idx,:] # pca_
        background = painting_attributes_[back_idx,:] # pca_
        n_target, _= target_set.shape
        n_backgr, _= background.shape
        back_covm = background.T.dot(background)/(n_backgr-1)
        targ_covm = target_set.T.dot(target_set)/(n_target-1)
        sigma = targ_covm - alpha*back_covm
        w, v = np.linalg.eig(sigma)
        eig_idx = np.argpartition(w, -2)[-2:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        v_top = v[:,eig_idx]
        reduced_dataset = target_set.dot(v_top)
        reduced_dataset[:,0] = reduced_dataset[:,0]*np.sign(reduced_dataset[0,0])
        reduced_dataset[:,1] = reduced_dataset[:,1]*np.sign(reduced_dataset[0,1])

        new_points = []
        for i,idx in enumerate(selected_idx):
            new_points.append({idx:reduced_dataset[i,:].tolist()})

        return flask.jsonify({'idx':selected_idx,'projection':new_points})
    else:
        return flask.jsonify([0])
    data = flask.request.get_json()
    selected_idx = data['data']
    label = data['label']
    alpha = data['alpha']
    back_idx = [i for i in range(painting_attributes.shape[0]) if i not in selected_idx]

    target_set = pca_[selected_idx,:]
    background = pca_[back_idx,:]
    n_target, _= target_set.shape
    n_backgr, _= background.shape

    back_covm = background.T.dot(background)/(n_backgr-1)
    targ_covm = target_set.T.dot(target_set)/(n_target-1)

    sigma = targ_covm - alpha*back_covm
    w, v = np.linalg.eig(sigma)
    print(w.shape,v.shape)
    eig_idx = np.argpartition(w, -2)[-2:]
    eig_idx = eig_idx[np.argsort(-w[eig_idx])]
    v_top = v[:,eig_idx]
    
    reduced_dataset = target_set.dot(v_top)
    reduced_dataset[:,0] = reduced_dataset[:,0]*np.sign(reduced_dataset[0,0])
    reduced_dataset[:,1] = reduced_dataset[:,1]*np.sign(reduced_dataset[0,1])
    #return reduced_dataset

    return flask.jsonify([0])
#

'''
TODO: run kmeans on painting_attributes, returning data in the same format as in the first part of the assignment. Namely, an array of objects containing the following properties:
    * label - the cluster label
    * id: the data item's id, simply its index
    * attribute: the attribute name
    * value: the binary attribute's value
'''
@app.route('/kmeans', methods=['GET'])
def kmeans():
    global km
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    painting_attributes_ = scaler.fit_transform(painting_attributes)
    km = KMeans(n_clusters=6,random_state=0).fit(painting_attributes_)
    labels = km.labels_
    kmeans_data = []
    for i in range(painting_attributes.shape[0]):
        for j in range(painting_attributes.shape[1]):
            kmeans_data.append({'attribute':attribute_names[j],
                                'id':int(i),
                                'label':labels[i].item(),
                                'value':painting_attributes[i,j].item()})
    print(0,km)
    return flask.jsonify(kmeans_data)
#

@app.route('/')
def home():
    return flask.render_template('index.html')

if __name__=='__main__':
    painting_image_urls = json.load(open('painting_image_urls.json','r'))
    attribute_names = json.load(open('attribute_names.json','r'))
    episode_names = json.load(open('episode_names.json','r'))
    painting_attributes = np.load('painting_attributes.npy')

    app.run(debug=True)
#
