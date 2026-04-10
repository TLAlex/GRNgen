library("igraph")
library("qgraph")
library("SEMgraph")

# Set working directory

setwd(".../BEELINE datasets/Synthetic/dyn-BF/dyn-BF-100-1")
getwd()

# import GroundTruth as edge list and dataset
edge_list <- read.csv(file = 'GroundTruth.csv')
dataset <- read.csv(file = 'ExpressionData.csv', row.names = 1)

# building the graph from edge_list

g <- graph_from_data_frame(edge_list, directed=TRUE, vertices=names(dataset))


#fit sem model
sem <- SEMrun(graph = g, data = dataset)

summary(sem$fit)

#Visualization
#par(mfrow=c(2,1))
gplot(sem$graph)

