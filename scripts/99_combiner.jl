using DrWatson
@quickactivate :RandomHALsims
using CSV
using DataFrames

dirs = ["testing"]

for dir in dirs
    dir_path = datadir(dir)
    
    # Get all CSV files in the directory
    metrics_files = filter(f -> endswith(f, "_metrics.csv"), readdir(dir_path))
    preds_files = filter(f -> endswith(f, "_preds.csv"), readdir(dir_path))
    
    # Read and combine all CSV files
    metrics = reduce(vcat, CSV.read(joinpath(dir_path, file), DataFrame) for file in metrics_files)
    preds = reduce(vcat, CSV.read(joinpath(dir_path, file), DataFrame) for file in preds_files)
    
    # Write combined CSV to output
    output_path_metrics = datadir("$(dir)-combined-metrics.csv")
    output_path_preds = datadir("$(dir)-combined-preds.csv")

    CSV.write(output_path_metrics, metrics)
    CSV.write(output_path_preds, preds)
end