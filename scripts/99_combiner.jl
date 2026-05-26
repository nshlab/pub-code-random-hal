using DrWatson
@quickactivate :RandomHALsims
using CSV
using DataFrames

dirs = ["3_small_comparison", "4_large_randomhal"]

for dir in dirs
    dir_path = datadir(dir)
    
    # Get all CSV files in the directory
    csv_files = filter(f -> endswith(f, ".csv"), readdir(dir_path))
    
    # Read and combine all CSV files
    dfs = [CSV.read(joinpath(dir_path, file), DataFrame) for file in csv_files]
    combined_df = vcat(dfs...)
    
    # Write combined CSV to output
    output_path = datadir("$(dir)-combined.csv")
    CSV.write(output_path, combined_df)
end