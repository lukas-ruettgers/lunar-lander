4 types of files for every run (total 7 runs):
1. "run_name".png file is for the chart and some metrics (you can use any of these info in the report for comparison)
2. "run_name".txt file is for the outputs (the score, avg_score, highscore for each episode)
***Note for 2 that in addition to the 3000 episodes, there is an evaluation performed at every 100 episodes, where the agent performs in a reset environment for 5 times, no learning, and an avg score out of the 5 is taken (this is represented by the red dots in the graph under point 1)
3. policygradient.h5 file is the weights of the agent (you can recreate the agent by loading in these weights into the keras model, no need to retrain)
4. "run_name".mp4 file is the clips of the lander (5 clips in total)
5. "run_name"mp4scores.txt file are the scores for the 5 recorded runs in the mp4
