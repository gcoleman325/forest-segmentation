library(TreeLS)
library(lidR)
library(nabor)

percent_unfilled_df <- data.frame()
voxel_metrics_df <- data.frame()

folder <- "D:/scans/all_preprocessed/ccb"
files <- list.files(pattern = ".las", folder)
scan_num = 0

for (i in 1:length(files)){
  scan_num <- scan_num + 1
  
  # read in file
  file <- files[i]
  file_name <- substring(file, 1, nchar(file)-4)
  tls <- readLAS(paste0(folder, "\\", file))
  
  strata_ranges <- list(
    strata1 = c(0, 0.5),
    strata2 = c(0.5, 1),
    strata3 = c(1, 1.5),
    strata4 = c(1.5, 2),
    strata5 = c(2, Inf)
  )
  
  for (strata_name in names(strata_ranges)) {
    range <- strata_ranges[[strata_name]]
    
    strata <- filter_poi(tls, Z >= range[1], Z < range[2])
    
    if (nrow(strata@data) > 0) {
      voxel_metrics <- voxel_metrics(strata, func = .stdmetrics, 0.05)
      
      unfilled_voxels <- sum(voxel_metrics$n <= 1)
      total_voxels <- nrow(voxel_metrics)
      percent_unfilled <- (unfilled_voxels / total_voxels) * 100
      row <- data.frame(range[1], percent_unfilled)
      percent_unfilled_df <- rbind(percent_unfilled_df, row)
      
      voxel_metrics_row <- data.frame(scan_num, range[1], voxel_metrics)
      voxel_metrics_df <- rbind(voxel_metrics_df, voxel_metrics_row)
    }
  }
}

# save results
write.csv(percent_unfilled_df, "D:/metrics/percent_unfilled_df.csv")
write.csv(voxel_metrics_df, "D:/metrics/voxel_metrics_df.csv")

