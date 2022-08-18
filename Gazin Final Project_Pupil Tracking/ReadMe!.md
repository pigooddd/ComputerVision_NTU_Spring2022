# ReadMe!
* This is the file tree for our zip file:
* Please put the "dataset" as the below format, thank you!

```
B09901073/
|-----README.md
|-----model_best.pt
|-----ganzin.py
|-----generate_result.py
|-----model.py
|-----run.sh
|-----dataset
|        |-----public
|        |        |-----S1
|        |        |-----S2
|        |        |-----S3
|        |        |-----S4
|        |-----test
|        |        |-----S5
|        |        |-----challenge set
|-----requirement.txt
```
* This is the step of execution:
* Noted: The method of executing the "challenge set" and "hidden set" is the same of the testing set. Just put the challenge set to the "test" folder as mentioned above.
* We will generate results for all the series in the "test" folder.
* The results will be placed in B09901073/S5_solution/ 
```
cd B09901073
```
```
bash run.sh
```
* To retrain the model, run
```
python ganzin.py
# model_best.pt will be overwrited by the new model
```
* To generate result with the current model_best.pt, simply run
* Same as above. All the series in dataset/test/ will be executed to generate result, and the result will be placed in S5_solution/
```
python generate_result.py
``` 

        