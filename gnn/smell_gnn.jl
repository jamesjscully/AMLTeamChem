using GeometricFlux, MolecularGraph, CSV, GraphSignals, DataFrames, Plots
using Statistics: mean
using LightGraphs:adjacency_matrix
using LightGraphs.SimpleGraphs
using GeometricFlux.Flux
using GeometricFlux.Flux: logitcrossentropy, @epochs
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
    adjmat = Array{Int32}(adjacency_matrix(g))

    fgraph = FeaturedGraph(adjmat, nodes, zeros(Float32, size(nodes)), [0f0])
    return (fgraph, molecular_properties)
end

train = get_features.(raw_train[!,1])
test = get_features.(raw_test[!,1])


## Encode Labels
using Flux: onehot, onecold

function encode(sentence)
    wordvec = Symbol.(split(sentence,","))
    convert.(Float32, sum(map(wordvec) do x
        Vector(onehot(x, vocab))
    end))
end
function decode(vec)
    n = length(vec)
    n = n < 3 ? n : 3
    I = sortperm(vec)[end-n+1:end]
    vocab[reverse(I)]
end

train_labels = map(raw_train[!,2]) do x
    encode(x)
end
#test encode/decode
#decode(train_labels[74])
## Create model


num_atom_features = size(train[1][1].nf, 1)
num_mol_features =  length(train[1][2])
vocablen = length(vocab)
heads = 4
hidden = 4
outfeatures = 7
σ1 = GeometricFlux.relu
σ2= GeometricFlux.sigmoid
negative_slope = .2f0

GATConv(num_atom_features => hidden, heads = heads; negative_slope = .2f0, σ = relu)

#graph network model
inner_model = Chain(
    GATConv(num_atom_features => hidden, heads = heads, negative_slope = negative_slope),
    GATConv(hidden*heads => outfeatures, heads = heads, concat = false, negative_slope = negative_slope),
    x -> sum.([x.nf[1,:] for i = 1:size(x.nf, 1)])
)

model = Chain(
    x -> vcat(inner_model(x[1]),x[2]),
    #Feedforward Network
    Dense(num_mol_features+outfeatures, vocablen),
    #Dense(vocablen, vocablen),
)

ps = params(model, inner_model)
#test model
## Loss
function loss(x,y)
    logits = model(x)
    sigcrossentropy = @. reduce(max,logits,init=0) - logits*y + log(1 + exp(-abs(logits)))
    return mean(sigcrossentropy)
end
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

data = GeometricFlux.Flux.Data.DataLoader(train, train_labels, batchsize = 100,
    shuffle = true, partial = true)

opt = ADAM()

jacard(train, raw_train[!,2])
loss(train, train_labels)

jacarr = Float64[]
lossarr = Float32[]

function my_train!(loss, ps, data, opt)
  for d in data
    # back is a method that computes the product of the gradient so far with its argument.
    train_loss, back = GeometricFlux.Zygote.pullback(() -> loss(d...), ps)
    # Insert whatever code you want here that needs training_loss, e.g. logging.
    println(train_loss)
    # logging_callback(training_loss)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    gs = back(one(train_loss))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    Flux.update!(opt, ps, gs)
    # Here you might like to check validation set accuracy, and break out to do early stopping.
  end
  @show l = loss(train, train_labels)
  @show jc = jacard(train, raw_train[!,2])
  push!(jacarr, jc)
  push!(lossarr, l)
  plot(lossarr) |> display
end
plot(jacarr)
@epochs 1000 my_train!(loss, ps, data, opt)

#speed test
"""inner_model = inner_model |> cpu
model = model |> cpu


import GeometricFlux.gpu
using GeometricFlux.CUDA

function gpu(x::FeaturedGraph{Array{Int32,2},Array{Float32,2},Array{Float32,2},Array{Float32,1}})
    FeaturedGraph(x.graph, cu(x.nf), cu(x.ef), cu(x.gf))
end
gpu(x::Tuple{FeaturedGraph,T} where T) = (cu(x[1]), cu(x[2]))

train_labels |> gpu
train = [(e[1] |>gpu, e[2] |> gpu) for e in train]

jacard(train, raw_train[!,2])
"""
