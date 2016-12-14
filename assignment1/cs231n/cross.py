
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(X_train,num_folds)
y_train_folds = np.array_split(y_train,num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    k_to_accuracies[k] = []

for k in k_choices:
    for m in range(num_folds):
        if m==0:
            xtrain_cv = np.concatenate(X_train_folds[1:])
            ytrain_cv = np.concatenate(y_train_folds[1:])
        elif m==4:
            xtrain_cv = np.concatenate(X_train_folds[0:m])
            ytrain_cv = np.concatenate(y_train_folds[0:m])
        else:
            xtrain_cv1 = np.concatenate(X_train_folds[0:m])
            xtrain_cv2 = np.concatenate(X_train_folds[m+1:])
            xtrain_cv = np.vstack((xtrain_cv1,xtrain_cv2))
            ytrain_cv1 = np.concatenate(y_train_folds[0:m])
            ytrain_cv2 = np.concatenate(y_train_folds[m+1:])
            ytrain_cv = np.hstack((ytrain_cv1,ytrain_cv2))
        xtest_cv = X_train_folds[m]
        ytest_cv = y_train_folds[m]

        num_test = ytest_cv.shape[0]

        classifier.train(xtrain_cv, ytrain_cv)
        dists = classifier.compute_distances_no_loops(xtest_cv)
        y_test_pred = classifier.predict_labels(dists,k)
        #classifier.train(xtrain_cv,ytrain_cv)
        #y_test_pred = classifier.predict(xtest_cv,k=k)
        num_correct = np.sum(y_test_pred == ytest_cv)
        accuracy = float(num_correct)/num_test
        k_to_accuracies[k].append(accuracy)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)











# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
learning_rates = [1e-7, 5e-5]
regularization_strengths = [5e4, 1e5]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################
for rate in learning_rates:
    for regu in regularization_strengths:
        results[(rate,regu)] = ()
for rate in learning_rates:
    for regu in regularization_strengths:
        svm = LinearSVM()
        loss_hist = svm.train(X_train, y_train, rate, regu,num_iters=1500, verbose=True)
        y_train_pred = svm.predict(X_train)
        y_val_pred = svm.predict(X_val)
        train_accuracy = np.mean(y_train == y_train_pred)
        val_accuracy = np.mean(y_val == y_val_pred)
        if val_accuracy>best_val:
            best_val = val_accuracy
            best_svm = svm
        results[(rate,regu)] = (train_accuracy,val_accuracy)
        print "results[(",rate,",",regu,")]:",results[(rate,regu)]

########################################11########################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)

print 'best validation accuracy achieved during cross-validation: %f' % best_val




# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.
from cs231n.classifiers import Softmax
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [5e4, 1e8]

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################
for rate in learning_rates:
    for regu in regularization_strengths:
        results[(rate,regu)] = ()
for rate in learning_rates:
    for regu in regularization_strengths:
        softmax = Softmax()
        loss_hist = softmax.train(X_train, y_train, rate, regu,num_iters=1500, verbose=True)
        y_train_pred = softmax.predict(X_train)
        y_val_pred = softmax.predict(X_val)
        train_accuracy = np.mean(y_train == y_train_pred)
        val_accuracy = np.mean(y_val == y_val_pred)
        if val_accuracy>best_val:
            best_softmax = softmax
            best_val = val_accuracy
        results[(rate,regu)] = (train_accuracy,val_accuracy)
        print "results[(",rate,",",regu,")]:",results[(rate,regu)]
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)









best_net = None # store the best model into this

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
input_size = 32*32*3
hidden_size = [50,100,150,200,250]
num_classes = 10
best_val = -1
best_net = None
for hidden in hidden_size:
    print "hidden_size:",hidden
    net = TwoLayerNet(input_size, hidden, num_classes)
    stats = net.train(X_train, y_train, X_val, y_val,num_iters=1000, batch_size=200,learning_rate=1e-4, learning_rate_decay=0.95,reg=0.5, verbose=True)
    val_acc = (net.predict(X_val) == y_val).mean()
    if val_acc>best_val:
        best_val = val_acc
        best_net = net
print "best validation accuracy:",val_acc

#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################

print 'best validation accuracy achieved during cross-validation: %f' % best_val






# Use the validation set to tune the learning rate and regularization strength

from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [1e5, 1e6, 1e7]

results = {}
best_val = -1
best_svm = None

for rate in learning_rates:
    for regu in regularization_strengths:
        results[(rate,regu)] = ()
################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
for rate in learning_rates:
    for regu in regularization_strengths:
        svm = LinearSVM()
        loss_hist = svm.train(X_train_feats, y_train, rate, regu,num_iters=1500, verbose=True)
        y_train_pred = svm.predict(X_train_feats)
        y_val_pred = svm.predict(X_val_feats)
        train_accuracy = np.mean(y_train == y_train_pred)
        val_accuracy = np.mean(y_val == y_val_pred)
        if val_accuracy>best_val:
            best_val = val_accuracy
            best_svm = svm
        results[(rate,regu)] = (train_accuracy,val_accuracy)
        print "results[(",rate,",",regu,")]:",results[(rate,regu)]
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)

print 'best validation accuracy achieved during cross-validation: %f' % best_val






from cs231n.classifiers.neural_net import TwoLayerNet

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None

################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
learning_rates = [1e-4, 1e-5, 1e-6]
regularization_strengths = [0.4,0.5, 0.6]
val_acc = -1
for rate in learning_rates:
    for regu in regularization_strengths:
        stats = net.train(X_train_feats, y_train,X_val_feats, y_val,num_iters=1000, batch_size=200,learning_rate=rate, learning_rate_decay=0.95,reg=regu, verbose=True)
        val_acc = (net.predict(,X_val_feats) == y_val).mean()
        if val_acc>best_val:
            best_val = val_acc
            best_net = net

print "best validation accuracy:",val_acc
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
