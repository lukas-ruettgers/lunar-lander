this was the original run.
as this was taking way too long and i noticed the scores dropping (past the optimal point), i was inspired to implement early stopping, 
which was used for the actual run 3 results (to great effect). you can mention this in the report if u want! (ask me for more info if u need)

#on initial running, it is observed that the average scores peaked early and only decreased with further training, which
#could be a symptom of overfitting (specifically, the model learned to hover in place, or continue firing and slowly moving 
#away from the target even after landing, which significantly impacted scores).
#to mitigate this, we changed the experiment setting. We will be training the model in rounds of 100 episodes each time 
#followed by an evaluation run over 10 episodes. We will be artificially stopping the training when the training score/
#average training score/evaluation score has reached around the 200 point mark and is observed to decrease (based on gut feel).
#this allows for more optimal training, given the circumstances. it also saves precious runtime (and reduce memory usage).
                                                                                                              
#we will also be using this approach for subsequent runs.
