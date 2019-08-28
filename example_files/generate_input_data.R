# SIMULATE TRAINING DATA
training_data = sample(1:4,1000,replace=T)

m1 = c(0,2,5,6)
s1 = c(1,5,2,.3)
e1 = c(0.2,0.8,0.05,1)
features = NULL
labels = NULL
for (i in training_data){
	features = rbind(features, c(rnorm(50, m1[i], s1[i]), rexp(50,e1[i]))    )
	labels = rbind(labels,c(i))
}

features = as.data.frame(features)
labels = as.data.frame(labels)



setwd("...")
write.table(features,file="training_features.txt",quote=F,sep="\t",row.names = F,col.names = F)
write.table(labels,file="training_labels.txt",quote=F,sep="\t",row.names = F,col.names = F)

# SIMULATE EMPIRICAL DATA
empirical_data = c(1,1,1,2,2,2,3,3,3,4,4,4)

m1 = c(0,2,5,6)
s1 = c(1,5,2,.3)
e1 = c(0.2,0.8,0.05,1)
features = NULL
for (i in empirical_data){
	features = rbind(features, c(rnorm(50, m1[i], s1[i]), rexp(50,e1[i]))    )
}

features = as.data.frame(features)

write.table(features,file="empirical_data.txt",quote=F,sep="\t",row.names = F,col.names = F)