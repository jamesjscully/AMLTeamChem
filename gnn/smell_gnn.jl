using GeometricFlux, MolecularGraph, CSV, Flux, GraphSignals, DataFrames, Plots
using Statistics: mean
using LightGraphs:adjacency_matrix
using LightGraphs.SimpleGraphs
using Flux: @epochs
using Flux: logitcrossentropy
#add ChemometricTools.jl features

## Get and clean data

#fn to get rid of smiles that do not parse to mol format
function clean(data)
    smiles = data.SMILES
    badidxs = []
    for i in eachindex(smiles)
        try
            smilestomol(smiles[i])
        catch
            push!(badidxs, i)
        end
    end
    return data[setdiff(1:end, badidxs),:]
end

raw_train = CSV.File("../dataset/train.csv") |> DataFrame |> clean
raw_test = CSV.File("../dataset/test.csv") |> DataFrame |> clean
_vocab = CSV.File("../dataset/unique_smells.csv") |> DataFrame
vocab = Array(Symbol.(_vocab))[:,1]
push!(vocab, :fruity)
## Select features to use for atom representation

function get_features(str)
    # convert smiles to molecule, and perform precalculations
    mol = smilestomol(str)
    precalculate!(mol)
    # fill nodes array with data related to each atom
    _nodes = []
    atoms = mol.nodeattrs
    for i in eachindex(atoms)
        atomfeatures = [
            atomnumber(atoms[i]), # element identity
            atoms[i].charge,
            atoms[i].isaromatic,
            exactmass(atoms[i])[1],
            mol.cache[:lonepair][i],
            mol.cache[:valence][i],
            mol.cache[:apparentvalence][i]
        ]
        atomfeatures[isnothing.(atomfeatures)] .= 0f0
        push!(_nodes, atomfeatures)
    end

    nodes = Array{Float32}(undef, length(_nodes[1]),length(_nodes))
    for (i, node) in enumerate(_nodes)
        nodes[:,i] = node
    end

    # get vector of global molecular_properties
    molecular_properties = Float32[
        try
            wclogp(mol)
        catch
            1.0
        end,
        exactmass(mol)[1],
        hacceptorcount(mol),
        hdonorcount(mol),
        rotatablecount(mol)
        # insert more from mordred
    ]
    # get array of edge features
    """
    _edges = []
    for e in mol.edgeattrs
        edgefeatures = [
            e.order,
            e.isaromatic ? 1. : 0.,
            (unspecified = 0., up = 1., down = -1.)[e.direction],
            (unspecified = 0., trans = 1., cis = -1.)[e.stereo],
        ]
        push!(_edges, edgefeatures)
    end
    if length(_edges) > 0
        edges = Array{Float32}(undef, length(_edges[1]),length(_edges))
        for (i, edge) in enumerate(_edges)
            edges[:,i] = edge
        end
    else
        edges = zeros(Float32, length(mol.edgeattrs))
    end
    """
    # generate adjacency_matrix and FeaturedGraph
    g = SimpleGraph(length(mol.nodeattrs))
    for e in mol.edges
        add_edge!(g,e...)
    end
    adjmat = adjacency_matrix(g)

    fgraph = FeaturedGraph(adjmat, nodes)
    return (fgraph, molecular_properties)
end

train = get_features.(raw_train[!,1])
test = get_features.(raw_test[!,1])


## Encode Labels
using Flux: onehot, onecold

function encode(sentence)
    wordvec = Symbol.(split(sentence,","))
    map(wordvec) do x
        onehot(x, vocab)
    end
end
function decode(vec)
    n = length(vec)
    n = n < 3 ? n : 3
    I = sortperm(vec)[end-n+1:end]
    vocab[reverse(I)]
end

train_labels = map(raw_train[!,2]) do x
    encode(x,vocab)
end
#test encode/decode
map(x -> decode(x)[1],train_labels[74])[1] = :fruity
## Create model

num_atom_features = size(train[1][1].nf, 1)
num_mol_features =  length(train[1][2])
vocablen = length(vocab)
heads = 3
hidden = 3
outfeatures = 7
σ1 = relu
σ2= leakyrelu

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
    softmax
)

ps = Flux.params(model, inner_model)

#test model
decode(model(train[8]), vocab)
## Loss
loss(x,y) = mean([logitcrossentropy(model(x),l) for l in y])
loss(x::AbstractArray,y::AbstractArray) = mean(loss.(x,y))

#test loss
#for i in eachindex(train)
#    loss(train[i], train_labels[i]) |> println
#    i |> println
#end

#_data = Flux.Data.DataLoader(train, train_labels, batchsize = 100,
#    shuffle = false, partial = true)
#for d in _data
#    loss(d...) |> println
#end

# test gradient
grad = gradient(() -> loss(train[1], train_labels[1]), ps)
[grad[e] for e in ps]

function jacard(X, Y)
    arr = []
    for (i,x) in enumerate(X)
        a = decode(model(x))
        b = Symbol.(split(Y[i],","))
        push!(arr, length(intersect(a,b))/ length(union(a,b)))
    end
    return mean(arr)
end
## Training

data = Flux.Data.DataLoader(train, train_labels, batchsize = 100,
    shuffle = true, partial = true)

opt = ADAM(0.05)

jacarr = []
function cb()
    jc = jacard(train, raw_train[!,2])
    push!(jacarr, jc)
    plot(jacarr)
end
cb()

@epochs 10 Flux.train!(loss, ps, data, opt, cb = cb)

#speed test
inner_model = inner_model |> cpu
model = model |> cpu

train = train |> cpu
train_labels = train_labels |> cpu
