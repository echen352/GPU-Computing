library(tidyverse)
library(dplyr)

s <- list(256, 512, 1024, 2048, 3072, 4096, 5120, 7680, 8192, 10240, 12800)
cuda_timings <- read_csv("results/cuda_timings.csv")
default_timings <- read_csv("results/default_timings.csv")

colnames(cuda_timings) <- c("size","time")
cdf <- data.frame(version=character(),size=integer(),time=double())

colnames(default_timings) <- c("size","time")
ddf <- data.frame(version=character(),size=integer(),time=double())

for (x in s) {
  res <- cuda_timings %>%
    filter(size == x) %>%
    group_by(size) %>%
    summarize(mean(time))
  
  cdf[nrow(cdf) + 1,] <- c("gpu", res)
}

for (x in s) {
  res <- default_timings %>%
    filter(size == x) %>%
    group_by(size) %>%
    summarize(mean(time))
  
  ddf[nrow(ddf) + 1,] <- c("default", res)
}

pdf <- rbind(cdf, ddf)
pdf$size <- as.character(pdf$size)

pdf %>%
  ggplot(aes(reorder(size,time), time, fill=version)) +
  geom_bar(stat="identity", position="dodge") +
  labs(title="Cuda vs Default Canny Edge Execution Times",
       x="Image Size", y="Time (sec)")

sdf <- data.frame(size=integer(), Ts=double(), Tp=double(), speedup=double())
for (x in s) {
  tp <- cdf %>% filter(size == x)
  ts <- ddf %>% filter(size == x)
  su <- ts$time / tp$time
  sdf[nrow(sdf) + 1,] <- c(x, ts$time, tp$time, su)
}
write.csv(sdf, file="results/speedup.csv")

sdf %>%
  ggplot(aes(reorder(size,speedup), speedup)) +
  geom_bar(stat="identity", position="dodge", fill="dodgerblue") +
  labs(title="Cuda vs Default Canny Edge Speedup",
       x="Image Size", y="Speedup")
