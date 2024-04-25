<h3> This is a cell detector Project to detect and slice super resolution images </h3>
<h3> It's built over pylibCZIrw lib to read and write high resolution CZI images and convert to png tiles  </h3>
<h3> Currently supports only Feulgen and Diff Quik Giemsa staining  </h3>
<h3> If you wan't more staining methods, change the color range in contains_cells function </h3>
<h3> Installation of this project </h3>


```sh
pip install git+https://github.com/LuisOtavioSantos/czi_image_tools.git
```
<h3> Or development mode </h3>

<h4> Create a virtual enviroment </h4>

```sh
python3 -m venv venv
```

<h4> Create a virtual enviroment </h4>

#### Ubuntu
```sh
python3 -m venv venv
```
#### Windows
```sh
python -m venv venv
```

<h4> Activate </h4>

#### Ubuntu

```sh
source venv/bin/activate
```

#### Windows
```sh

.\venv\bin\activate
```

<h4> install libraries </h4>

#### Ubuntu\windows
```sh
pip install -r requirements.pip
```





