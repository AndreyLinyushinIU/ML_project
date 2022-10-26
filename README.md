## Running the code

From the project directory:

* ```mkdir models/pretrained``` and copy inside it pretrained model from ML assignment 1.
* You should have <em>config.yml</em> file at project root directory containing bot token (see <em>config-example.yml</em>).
* ```python -m bot``` - process, supporting bot.
* Download and copy <em>decoder_epoch_20.pth</em>, <em>decoder_epoch_20.pth</em> from 
http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/ into DeepPhotoStyle_pytorch/seg_checkpoint
 
The bot is accessible at Telegram: <em>@style_transferer_bot</em>, start it with <em>/run</em> command.
