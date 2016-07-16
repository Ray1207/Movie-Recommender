# A Machine Learning Based Movie Recommender System

In this project, I implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings.
The movie dataset consists of ratings on a scale of 1 to 5. The dataset has nu = 943 users, and nm = 1682 movies.
(http://grouplens.org/datasets/movielens/)

The goal of this project is to offer its users recommendations of movies they might like - like Netflix. 
These recommendations are based on the historical ratings provided by users.

## Installation

This project is implemented in Octave (compatible with Matlab), a high-level programming
language well-suited for numerical computations. If you do not have
Octave or Matlab installed, please refer to the installation instructions on the offical Octave or Matlab websites.

## Usage

Step 1: Open the Recommender project folder in Octave

Step 2: Open file main.m, go to Step 2 - here you can enter your own movie preferences, so that later when the algorithm
runs, you can get your own movie recommendations. I have filled out
some values according to my own preferences, but you should change this
according to your own tastes. The list of all movies and their number in the
dataset can be found listed in the file movie ids.txt.

Step 3: In Octave, type command - "main". Now the system will start training the collaborative filtering algorithm
to make movie recommendations for you.

## Techniques Used

1. Collaborative Filtering Model (with L2 Regularisation)
2. Gradient Descent
3. Vectorisation Programming

