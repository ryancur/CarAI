import React, { useState, useEffect } from 'react';

function PredictAutoPriceForm() {

    // initialize list for drop down selector  
    // const [models, setModels] = useState([]);
    // const [manufacturers, setManufacturers] = useState([]);

    // initialize variables with state
    const [color, setColor] = useState('');
    const [year, setYear] = useState('');
    const [vin, setVin] = useState('');
    const [selectedModel, setSelectedModel] = useState('');
    const [selectedManufacturer, setSelectedManufacturer] = useState('');

    // initialize variables for display
    const [dColor, setDColor] = useState('');
    const [dYear, setDYear] = useState('');
    const [dVin, setDVin] = useState('');
    const [dSelectedModel, setDSelectedModel] = useState('');
    const [dSelectedManufacturer, setDSelectedManufacturer] = useState('');
    const [dSuggestedPrice, setDSuggestedPrice] = useState('');

    // event handlers when changes are made to the form
    const handleColorChange = (event) => {
        const value = event.target.value;
        setColor(value);
    };

    const handleYearChange = (event) => {
        const value = event.target.value;
        setYear(value);
    };

    const handleVinChange = (event) => {
        const value = event.target.value;
        setVin(value);
    };

    const handleSelectedModelChange = (event) => {
        const value = event.target.value;
        setSelectedModel(value);
    };

    const handleSelectedManufacturerChange = (event) => {
        const value = event.target.value;
        setSelectedManufacturer(value);
    };

    // handle the data once submitted 
    const handleSubmit = async (event) => {
        event.preventDefault();
        const data = {};

        data.color = color;
        data.year = year;
        data.vin = vin;
        data.model = selectedModel;
        data.manufacturer = selectedManufacturer;

        console.log("Sending Data: ", data);


        const pricePredictionUrl = 'http://localhost:8070/api/automobile/predict-price/';
        const fetchConfig = {
            method: "post",
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json',
            }
        };

        const response = await fetch(pricePredictionUrl, fetchConfig);

        if (response.ok) {

            const responseData = await response.json();
            console.log("Response Data: ", JSON.stringify(responseData));

            setColor('');
            setYear('');
            setVin('');
            setSelectedModel('');
            setSelectedManufacturer('');

            setDColor(responseData.auto.color);
            setDYear(responseData.auto.year);
            setDVin(responseData.auto.vin);
            setDSelectedModel(responseData.auto.model);
            setDSelectedManufacturer(responseData.auto.manufacturer);
            setDSuggestedPrice(responseData.auto.suggested_price);
        };
    };

    // fetch data for selector 
    // const fetchData = async () => {
    //     const modelUrl = 'http://localhost:8100/api/models/';
    //     const modelResponse = await fetch(modelUrl);
    //     if (modelResponse.ok) {
    //         const modelData = await modelResponse.json();
    //         setModels(modelData.models);
    //     }
    //     const manufacturerUrl = 'http://localhost:8100/api/manufacturers/';
    //     const manufacturerResponse = await fetch(manufacturerUrl);
    //     if (manufacturerResponse.ok) {
    //         const manufacturerData = await manufacturerResponse.json();
    //         setManufacturers(manufacturerData.manufacturers);
    //     }
    // };

    // useEffect(() => {
    //     fetchData();
    // }, []);


    return (
        <div className="row">
            <div className="offset-3 col-6">
                <div className="shadow p-4 mt-4">
                    <h1>Price an Automobile</h1>
                    <form onSubmit={handleSubmit} id="price-automobile-form">
                        <div className="form-floating mb-3">
                            <input onChange={handleColorChange} value={color} placeholder="Color..." required type="text" name="color" id="color" className="form-control" />
                            <label htmlFor="color">Color</label>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handleYearChange} value={year} placeholder="Year..." required type="text" name="year" id="year" className="form-control" />
                            <label htmlFor="year">Year</label>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handleVinChange} value={vin} placeholder="VIN..." type="text" name="vin" id="vin" className="form-control" />
                            <label htmlFor="vin">VIN</label>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handleSelectedManufacturerChange} value={selectedManufacturer} placeholder="Manufacturer..." type="text" name="manufacturer" id="manufacturer" className="form-control" />
                            <label htmlFor="manufacturer">Manufacturer</label>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handleSelectedModelChange} value={selectedModel} placeholder="Model..." type="text" name="model" id="model" className="form-control" />
                            <label htmlFor="model">Model</label>
                        </div>
                        {/* <div className="mb-3">
                            <label htmlFor="model">Model</label>
                            <select onChange={handleSelectedModelChange} value={selectedModel} required name="model" id="model" className="form-control">
                                <option value="">Choose a model...</option>
                                {models.map(model => {
                                    return (
                                        <option key={model.id} value={model.name}>
                                            {model.name}
                                        </option>
                                    )
                                })}
                            </select>
                        </div>
                        <div className="mb-3">
                            <label htmlFor="manufacturer">Manufacturer</label>
                            <select onChange={handleSelectedManufacturerChange} value={selectedManufacturer} required name="manufacturer" id="manufacturer" className="form-control">
                                <option value="">Choose a manufacturer...</option>
                                {manufacturers.map(manufacturer => {
                                    return (
                                        <option key={manufacturer.id} value={manufacturer.name}>
                                            {manufacturer.name}
                                        </option>
                                    )
                                })}
                            </select>
                        </div> */}
                        <button className="btn btn-primary">Get Price</button>
                    </form>
                </div>
            </div>
            <div>
                <center>
                    <h2>Suggested Price</h2>
                    <h3>{dSuggestedPrice}</h3>
                    <p>Color: {dColor}</p>
                    <p>Year: {dYear}</p>
                    <p>VIN: {dVin}</p>
                    <p>Manufacturer: {dSelectedManufacturer}</p>
                    <p>Model: {dSelectedModel}</p>
                </center>
            </div >
        </div >
    );
}

export default PredictAutoPriceForm;