using Distributed

addprocs(2)


##

A = rand(1000,1000);
Bref = @spawnat :any A^2;

##