There are now two high-level directories. 
One contains the code to build a useable file (cleaned) from the raw inputs. 

The other contains code to take the cleaned data and turn it into figures and tables for the analysis.

First of all install git bash and create a repository on git

1) Go to working directory where the file exist which you want to push on remote and create .git folder by

$ git init

2) Add the files in your new local repository. $ git add . Note: while you are in same folder make sure you have placed dot after command if you putting path or not putting dot that will create ambiguity

3) Commit the files that you've staged in your local repository. $ git commit -m "First commit"

4) after this go to git repository and copy remote URL $ git remote add origin remote repository URL

5) $ git remote -v Note: this will ask for user.email and user.name just put it as per config

6) $ git push origin master this will push whole committed code to FILE.git on repository
