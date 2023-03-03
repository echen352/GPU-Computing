#data <- read.csv("timings_updated.csv")
#data$threads <- as.character(data$threads)
#ggplot(data, aes(size, time, color = threads)) +
#  geom_line() +
#  geom_point() +
#  labs(title = "Execution Time over Image Size and Thread Count") +
#  xlab("image size (height px)") +
#  ylab("execution time (us)")

data <- read.csv("timings_updated.csv")
data$threads <- as.character(data$threads)
data$size <- as.character(data$size)
ggplot(data, aes(threads, time, color = size)) +
  geom_point() +
  labs(title = "Execution Time over Image Size and Thread Count") +
  xlab("thread count") +
  ylab("execution time (us)")
ggsave("timings_plot_updated.png")