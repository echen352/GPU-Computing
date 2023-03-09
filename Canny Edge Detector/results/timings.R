data <- read_csv("timingscsv_averaged.csv")
data %>%
  ggplot(aes(threads, time, color = size)) +
  geom_point() +
  ylab("time (us)")
