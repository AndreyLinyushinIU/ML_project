## Running the code

From the project directory:

* ```mkdir output```
* ```mkdir pretrained-model``` and copy inside it pretrained model
* ```python nst_generate.py monet.jpg  stone_style.jpg monet-stone_style.jpg --num_iterations 200 --learning_rate 2 --save_amount 20```
 
First three parameters are input content image, input style image and name of resulting image. Other parameters are optional.