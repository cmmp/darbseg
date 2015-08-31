Sys.setenv(JAVA_HOME="") # have to do this for rJava...
setwd("C:/Users/Cássio/Dropbox/workspace/darbseg")

require(rJava)

.jinit()
.jaddClassPath("C:/Users/Cássio/Dropbox/workspace/darbseg/target/darbseg-0.1-jar-with-dependencies.jar")

X = as.matrix(read.table("C:/Users/Cássio/Dropbox/workspace/darbseg/resources/Y2.csv"))
#X = as.matrix(read.table("C:/Users/Cássio/Dropbox/workspace/darbseg/resources/sample.csv"))
# X = as.matrix(read.table("C:/Users/Cássio/Dropbox/workspace/darbseg/resources/sample2.csv"))
 # X = as.matrix(read.table("C:/Users/Cássio/Dropbox/workspace/darbseg/resources/sample3.csv"))

# set.seed(1234)
# Y1 = matrix(rnorm(30*2, mean = 0, sd = 1), 30, 2)
# Y2 = matrix(rnorm(30*2, mean = 10, sd = 1), 30, 2)
# X = rbind(Y1, Y2)
#write.table(X, "resources/sample3.csv", quote = FALSE, col.names = FALSE, row.names = F, sep = ' ')

darb = J("br.fapesp.darbseg.DarbellayUniformSegmentation")

lbls = darb$darbellay(.jarray(X, dispatch = TRUE))

plot(X, col = as.factor(lbls))
legend('bottomright', legend = unique(lbls), col = 1:length(lbls), pch = 1)



