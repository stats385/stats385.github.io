

```python
import matplotlib as mpl
import networkx as nx
import visJS2jupyter.visJS_module
import visJS2jupyter.visualizations as visualizations

G = nx.Graph()
```


```python
# nodes

# topics
topics = ["Approximation Theory",
          "Harmonic Analysis",
          "Neuroscience",
          "Statistics/Machine Learning",
          "Optimization"]
G.add_nodes_from(topics)

# questions
questions = ["Overfitting",
             "Regularization",
             "Implicit regularization",
             "Generalization",
             "Curse of dimensionality",
             "Deep versus shallow",
             "Landscape of loss",
             "Biological plausibility",
             "Robustness to noise",
             "Stability to deformations",
             "Energy propagation",
             "Translation invariance",
             "Invariance to intraclass variability",
             "Generative modeling",
             "Architectures as inference algorithms",
             "Interpretation of tricks",
             "Sparse coding",
             "Network design"]
G.add_nodes_from(questions)

# lectures in the stats385 course
lectures = ["Lecture 3:\nHelmut Bolcskei",
            "Lecture 4:\nAnkit Patel",
            "Lecture 5:\nTomaso Poggio"]
G.add_nodes_from(lectures)

nodes = G.nodes()

nodes_shape = []
node_shape = ['square' if (node in topics) else ('dot' if (node in questions) else 'star') for node in G.nodes()]
node_to_shape = dict(zip(G.nodes(),node_shape))

nodes_size = []
node_size = [5 if (node in topics) else (4 if (node in questions) else 3) for node in G.nodes()]
node_to_size = dict(zip(G.nodes(),node_size))

node_to_color = []
node_color = ['#800080' if (node in topics) else ('#C70039' if (node in questions) else '#FFC300') for node in G.nodes()]
node_to_color = dict(zip(G.nodes(),node_color))
```


```python
# edges

# topics and questions
topics_questions = [("Approximation Theory","Curse of dimensionality",{'title':"How does deep learning overcome the curse of dimensionality?"}),
                    ("Approximation Theory","Deep versus shallow",{'title':"When and why are deep networks better than shallow networks?"}),
                    ("Harmonic Analysis","Translation invariance",{'title':"Are the features computed by deep architectures translation invariant or covariant?"}),
                    ("Harmonic Analysis","Stability to deformations",{'title':"Are the features stable to deformations?"}),
                    ("Harmonic Analysis","Energy propagation",{'title':"How does the energy of the features behave as function of the layer and can we tune it?"}),
                    ("Harmonic Analysis","Robustness to noise",{'title':"If noise is added to the input of the architecture, do the estimated feature representations remain stable?"}),
                    ("Harmonic Analysis","Invariance to intraclass variability",{'title':"Is the network learning invariance to certain variabilities?"}),
                    ("Harmonic Analysis","Network design",{'title':"How should we design the architecture?"}),
                    ("Neuroscience","Biological plausibility",{'title':"Are modern architectures biologically plausible?"}),
                    ("Statistics/Machine Learning","Overfitting",{'title':"How can deep learning not overfit?"}),
                    ("Statistics/Machine Learning","Regularization",{'title':"How does the network manage to generalize without regularization despite the large number of parameters?"}),
                    ("Statistics/Machine Learning","Implicit regularization",{'title':"Is there any implicit regularization in the network?"}),
                    ("Statistics/Machine Learning","Generalization",{'title':"What does the generalization capability of a network rely on?"}),
                    ("Statistics/Machine Learning","Generative modeling",{'title':"Is there a generative model behind deep learning architectures and if so what is it?"}),
                    ("Statistics/Machine Learning","Architectures as inference algorithms",{'title':"Can we interpret certain networks as an inference algorithm under a generative model?"}),
                    ("Statistics/Machine Learning","Interpretation of tricks",{'title':"What is the importance of different tricks?"}),
                    ("Statistics/Machine Learning","Sparse coding",{'title':"Why do the filters trained in the first layer resemble those obtained from sparse coding?"}),
                    ("Optimization","Landscape of loss",{'title':"What is the landscape of the objective function being minimized?"})]
G.add_edges_from(topics_questions)

# lectures and questions
lectures_questions = [("Lecture 3:\nHelmut Bolcskei","Translation invariance",{'title':"Features become more translation invariant with increasing network depth."}),
                      ("Lecture 3:\nHelmut Bolcskei","Stability to deformations",{'title':"For certain classes of signals, the features are stable to deformations."}),
                      ("Lecture 3:\nHelmut Bolcskei","Energy propagation",{'title':"The energy of the features decays exponentially or polynomially, depending on the assumptions on the filters."}),
                      ("Lecture 3:\nHelmut Bolcskei","Network design",{'title':"Filters or number of layers can be designed for energy preservation."}),
                      ("Lecture 3:\nHelmut Bolcskei","Deep versus shallow",{'title':"Depth width tradeoff for networks with wavelet filters."}),
                      ("Lecture 4:\nAnkit Patel","Generative modeling",{'title':"NN-DRMM: a heirarchical generative deep sparse coding model."}),
                      ("Lecture 4:\nAnkit Patel","Architectures as inference algorithms",{'title':"Convnets are a max-sum-product message passing bottom-up inference algorithm."}),
                      ("Lecture 4:\nAnkit Patel","Interpretation of tricks",{'title':"ReLU and max pooling as max-marginalization."}),
                      ("Lecture 4:\nAnkit Patel","Sparse coding",{'title':"Only the sparse set of active paths matter for the final decision of the convnet."}),
                      ("Lecture 5:\nTomaso Poggio","Deep versus shallow",{'title':"For compositional functions deep networks avoid the curse of dimensionality because of locality of constituent functions."}),
                      ("Lecture 5:\nTomaso Poggio","Landscape of loss",{'title':"Many global minima that are found by SGD with high probability."}),
                      ("Lecture 5:\nTomaso Poggio","Overfitting",{'title':"Gradient descent avoids overfitting without explicit regularization, despite overparametrization."}),
                      ("Lecture 5:\nTomaso Poggio","Implicit regularization",{'title':"Gradient descent results in implicit regularization in deep linear networks."})]
G.add_edges_from(lectures_questions)

edges = G.edges(data=True)

edge_to_color = []
edge_color = ['gray' if (edge in topics_questions) else 'gray' for edge in G.edges()]
edge_to_color = dict(zip(G.edges(),edge_color))
```


```python
# set node initial positions using networkx's spring_layout function
pos = nx.spring_layout(G)

nodes_dict = [{"id":n,
               "color":node_to_color[n],
               "node_shape":node_to_shape[n],
               "degree":node_to_size[n],
               "x":pos[n][0]*1000,
               "y":pos[n][1]*1000} for n in nodes]
node_map = dict(zip(nodes,range(len(nodes))))  # map to indices for source/target in edges
edges_dict = [{"source":node_map[edges[i][0]], "target":node_map[edges[i][1]], 
              "color":"gray",
              "title":edges[i][2]['title']} for i in range(len(edges))]

# set some network-wide styles
visJS2jupyter.visJS_module.visjs_network(nodes_dict,edges_dict,
                          node_size_multiplier=10,
                          node_size_transform = '',
                          node_color_highlight_border='black',
                          node_color_highlight_background='#8BADD3',
                          node_color_hover_border='blue',
                          node_color_hover_background='#8BADD3',
                          node_font_size=20,
                          edge_arrow_to=False,
                          physics_enabled=True,
                          edge_color_highlight='#8BADD3',
                          edge_color_hover='#8BADD3',
                          edge_width=3,
                          edge_title_field='title',
                          max_velocity=15,
                          min_velocity=1)
```




<!doctype html><html><head>  <title>Network | Basic usage</title></head><body><script type="text/javascript">function setUpFrame() {     var frame = window.frames["style_file0"];    frame.runVis([{"id": "Approximation Theory", "color": "#800080", "node_shape": "square", "degree": 5, "x": 875.546614910906, "y": 176.7073251949794, "border_width": 0, "title": "Approximation Theory"}, {"id": "Harmonic Analysis", "color": "#800080", "node_shape": "square", "degree": 5, "x": 208.41844922354107, "y": 198.5350629305192, "border_width": 0, "title": "Harmonic Analysis"}, {"id": "Neuroscience", "color": "#800080", "node_shape": "square", "degree": 5, "x": 212.35524801346332, "y": 88.22913197018072, "border_width": 0, "title": "Neuroscience"}, {"id": "Statistics/Machine Learning", "color": "#800080", "node_shape": "square", "degree": 5, "x": 581.6843662642203, "y": 823.6879400495153, "border_width": 0, "title": "Statistics/Machine Learning"}, {"id": "Optimization", "color": "#800080", "node_shape": "square", "degree": 5, "x": 729.4773970219335, "y": 940.3313511622614, "border_width": 0, "title": "Optimization"}, {"id": "Overfitting", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 256.08743236858976, "y": 814.6448157979535, "border_width": 0, "title": "Overfitting"}, {"id": "Regularization", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 170.44299826740394, "y": 869.9801752538561, "border_width": 0, "title": "Regularization"}, {"id": "Implicit regularization", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 303.325568191846, "y": 958.3715232367283, "border_width": 0, "title": "Implicit regularization"}, {"id": "Generalization", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 533.8960536711354, "y": 1000.0, "border_width": 0, "title": "Generalization"}, {"id": "Curse of dimensionality", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 959.3083093120307, "y": 307.5867672112002, "border_width": 0, "title": "Curse of dimensionality"}, {"id": "Deep versus shallow", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 623.8294247799863, "y": 349.84762145320394, "border_width": 0, "title": "Deep versus shallow"}, {"id": "Landscape of loss", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 432.8631517592568, "y": 991.8228446104232, "border_width": 0, "title": "Landscape of loss"}, {"id": "Biological plausibility", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 482.9722177299119, "y": 0.0, "border_width": 0, "title": "Biological plausibility"}, {"id": "Robustness to noise", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 23.934254009514444, "y": 346.62441594072175, "border_width": 0, "title": "Robustness to noise"}, {"id": "Stability to deformations", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 376.1669616970537, "y": 14.298661164802223, "border_width": 0, "title": "Stability to deformations"}, {"id": "Energy propagation", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 288.4333293114901, "y": 51.766733919893554, "border_width": 0, "title": "Energy propagation"}, {"id": "Translation invariance", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 99.37178775966095, "y": 201.9558885658802, "border_width": 0, "title": "Translation invariance"}, {"id": "Invariance to intraclass variability", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 0.0, "y": 467.7663852751535, "border_width": 0, "title": "Invariance to intraclass variability"}, {"id": "Generative modeling", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 661.3178770661024, "y": 971.7602371303534, "border_width": 0, "title": "Generative modeling"}, {"id": "Architectures as inference algorithms", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 906.6894072927827, "y": 781.7497790218408, "border_width": 0, "title": "Architectures as inference algorithms"}, {"id": "Interpretation of tricks", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 962.4359888939355, "y": 676.3519374985898, "border_width": 0, "title": "Interpretation of tricks"}, {"id": "Sparse coding", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 853.5827192043517, "y": 676.4297287559405, "border_width": 0, "title": "Sparse coding"}, {"id": "Network design", "color": "#C70039", "node_shape": "dot", "degree": 4, "x": 578.0076486727472, "y": 14.325889131567804, "border_width": 0, "title": "Network design"}, {"id": "Lecture 3:\nHelmut Bolcskei", "color": "#FFC300", "node_shape": "star", "degree": 3, "x": 425.71191956759225, "y": 147.67543833614388, "border_width": 0, "title": "Lecture 3:\nHelmut Bolcskei"}, {"id": "Lecture 4:\nAnkit Patel", "color": "#FFC300", "node_shape": "star", "degree": 3, "x": 853.739903062142, "y": 849.5677845848898, "border_width": 0, "title": "Lecture 4:\nAnkit Patel"}, {"id": "Lecture 5:\nTomaso Poggio", "color": "#FFC300", "node_shape": "star", "degree": 3, "x": 398.3023957605418, "y": 702.0736304622405, "border_width": 0, "title": "Lecture 5:\nTomaso Poggio"}], [{"source": 0, "target": 9, "color": "gray", "title": "How does deep learning overcome the curse of dimensionality?"}, {"source": 0, "target": 10, "color": "gray", "title": "When and why are deep networks better than shallow networks?"}, {"source": 1, "target": 16, "color": "gray", "title": "Are the features computed by deep architectures translation invariant or covariant?"}, {"source": 1, "target": 14, "color": "gray", "title": "Are the features stable to deformations?"}, {"source": 1, "target": 15, "color": "gray", "title": "How does the energy of the features behave as function of the layer and can we tune it?"}, {"source": 1, "target": 13, "color": "gray", "title": "If noise is added to the input of the architecture, do the estimated feature representations remain stable?"}, {"source": 1, "target": 17, "color": "gray", "title": "Is the network learning invariance to certain variabilities?"}, {"source": 1, "target": 22, "color": "gray", "title": "How should we design the architecture?"}, {"source": 2, "target": 12, "color": "gray", "title": "Are modern architectures biologically plausible?"}, {"source": 3, "target": 5, "color": "gray", "title": "How can deep learning not overfit?"}, {"source": 3, "target": 6, "color": "gray", "title": "How does the network manage to generalize without regularization despite the large number of parameters?"}, {"source": 3, "target": 7, "color": "gray", "title": "Is there any implicit regularization in the network?"}, {"source": 3, "target": 8, "color": "gray", "title": "What does the generalization capability of a network rely on?"}, {"source": 3, "target": 18, "color": "gray", "title": "Is there a generative model behind deep learning architectures and if so what is it?"}, {"source": 3, "target": 19, "color": "gray", "title": "Can we interpret certain networks as an inference algorithm under a generative model?"}, {"source": 3, "target": 20, "color": "gray", "title": "What is the importance of different tricks?"}, {"source": 3, "target": 21, "color": "gray", "title": "Why do the filters trained in the first layer resemble those obtained from sparse coding?"}, {"source": 4, "target": 11, "color": "gray", "title": "What is the landscape of the objective function being minimized?"}, {"source": 5, "target": 25, "color": "gray", "title": "Gradient descent avoids overfitting without explicit regularization, despite overparametrization."}, {"source": 7, "target": 25, "color": "gray", "title": "Gradient descent results in implicit regularization in deep linear networks."}, {"source": 10, "target": 23, "color": "gray", "title": "Depth width tradeoff for networks with wavelet filters."}, {"source": 10, "target": 25, "color": "gray", "title": "For compositional functions deep networks avoid the curse of dimensionality because of locality of constituent functions."}, {"source": 11, "target": 25, "color": "gray", "title": "Many global minima that are found by SGD with high probability."}, {"source": 14, "target": 23, "color": "gray", "title": "For certain classes of signals, the features are stable to deformations."}, {"source": 15, "target": 23, "color": "gray", "title": "The energy of the features decays exponentially or polynomially, depending on the assumptions on the filters."}, {"source": 16, "target": 23, "color": "gray", "title": "Features become more translation invariant with increasing network depth."}, {"source": 18, "target": 24, "color": "gray", "title": "NN-DRMM: a heirarchical generative deep sparse coding model."}, {"source": 19, "target": 24, "color": "gray", "title": "Convnets are a max-sum-product message passing bottom-up inference algorithm."}, {"source": 20, "target": 24, "color": "gray", "title": "ReLU and max pooling as max-marginalization."}, {"source": 21, "target": 24, "color": "gray", "title": "Only the sparse set of active paths matter for the final decision of the convnet."}, {"source": 22, "target": 23, "color": "gray", "title": "Filters or number of layers can be designed for energy preservation."}]);}</script><iframe name="style_file0" src="style_file0.html" height="1200px" width="100%;"></iframe></body></html>


