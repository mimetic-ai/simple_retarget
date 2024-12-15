Retargeting is the process of recreating a motion performed by some demonstrator with another entity, which can be an animated figure or a robot. In this case, we are retargetting human motions onto a robot 
to allow for robot learning from demonstration. 

The approach here was to perform simple gradient descent based optimization. The human motion was represented by tracking the location of the elbow and wrist joint, treating the shoulder to elbow and elbow to wrist 
bones as rigid bodies. 

The retargeting works by performing gradient descent to move the robot joint angles in the direction needed to minimize the difference between relative location of robot elbow with relative location of human elbow, and 
relative location of robot wrist with relative location of human wrist. 
