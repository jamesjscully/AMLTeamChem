
num_atom_features = size(train[1][1].nf, 1)
num_mol_features =  length(train[1][2])
vocablen = length(vocab)
heads = 4
hidden = 4
outfeatures = 7
σ1 = GeometricFlux.relu
σ2= GeometricFlux.sigmoid

#graph network model
inner_model = Chain(
    GATConv(num_atom_features => hidden, heads = heads),
    GATConv(hidden*heads => outfeatures, heads = heads, concat = false),
    x -> sum.([x.nf[1,:] for i = 1:size(x.nf, 1)])
)

model = Chain(
    x -> vcat(inner_model(x[1]),x[2]),
    #Feedforward Network
    Dense(num_mol_features+outfeatures, vocablen, σ2),
    #Dense(vocablen, vocablen),
    softmax
)

ps = params(model, inner_model)


##TODO
#test only gnn
#add lots of molecular features
#try to speed up
#plot loss curve
#cross validation
num_atom_features = size(train[1][1].nf, 1)
num_mol_features =  length(train[1][2])
vocablen = length(vocab)
#heads = 4
hidden = 16
outfeatures = 20
σ1 = GeometricFlux.relu
σ2= GeometricFlux.sigmoid

#graph network model
inner_model = Chain(
    GCNConv(num_atom_features => hidden, σ1),
    GCNConv(hidden => outfeatures, σ1),
    x -> sum.([x.nf[1,:] for i = 1:size(x.nf, 1)])
)

outdims = length(inner_model(train[1][1]))

model = Chain(
    x -> vcat(inner_model(x[1]),x[2]),
    #Feedforward Network
    Dense(num_mol_features+outdims, vocablen, σ2),
    #Dense(vocablen, vocablen),
    softmax
)
