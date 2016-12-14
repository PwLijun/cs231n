
# TODO: Use a five-layer Net to overfit 50 training examples.

num_train = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

learning_rate = 1e-1
weight_scale = 1e-4
def getlearn(learning_rate,weight_scale):
    model = FullyConnectedNet([100, 100, 100, 100],
                    weight_scale=weight_scale, dtype=np.float64)
    solver = Solver(model, small_data,
                    print_every=10, num_epochs=20, batch_size=25,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': learning_rate,
                    }
             )
    solver.train()
    return solver.train_acc_history

m = True
while m:
    learning_rate = np.random.uniform(1e-5,1e-1)
    weight_scale = np.random.uniform(1e-5,1e-1)
    train_acc = getlearn(learning_rate,weight_scale)
    if max(train_acc) ==1.0:
        m = False
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()



best_model = None
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
################################################################################
small_data = {
  'X_train': data['X_train'],
  'y_train': data['y_train'],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}
learning_rates= = {}
update_rules = ['adam','rmsprop','sgd']
def getbest(update_rule,learning_rate):
    print 'running with ', update_rule
    model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)
    solver = Solver(model, small_data,
                  num_epochs=5, batch_size=100,
                  update_rule=update_rule,
                  optim_config={
                    'learning_rate': learning_rate
                  },verbose=True)
    solver.train()
    return model,solver.train_acc_history

m=True
while m:
    for update_rule in update_rules:
        learning_rate = np.random.uniform(1e-5,1e-1)
        learning_rates[update_rule] = learning_rate
        model,train_acc = getbest(update_rule,learning_rate)
        if train_acc >=0.55:
            m = False
            best_model = model



################################################################################
#                              END OF YOUR CODE                                #
################################################################################
