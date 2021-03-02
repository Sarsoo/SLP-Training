%% load and export dataset
[cancerInputs, cancerTargets] = cancer_dataset;

writematrix(cancerInputs, 'features.csv')
writematrix(cancerTargets, 'targets.csv')