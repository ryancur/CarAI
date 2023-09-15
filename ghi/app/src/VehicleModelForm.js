import React, { useState, useEffect } from 'react';


function VehicleModelForm() {

    const [manufacturers, setManufacturers] = useState([]);

    const [modelName, setModelName] = useState('');

    const [pictureUrl, setPictureUrl] = useState('');

    const [selectedManufacturer, setSelectedManufacturer] = useState('');

    const handleModelNameChange = (event) => {
        const value = event.target.value;
        setModelName(value);
    }

    const handlePictureUrlChange = (event) => {
        const value = event.target.value;
        setPictureUrl(value);
    }

    const handleSelectedManufacturerChange = (event) => {
        const value = event.target.value;
        setSelectedManufacturer(value);
    }


    const handleSubmit = async (event) => {
        event.preventDefault();
        const data = {};
        data.name = modelName;
        data.picture_url = pictureUrl;
        data.manufacturer_id = selectedManufacturer;

        const vehicleModelUrl = 'http://localhost:8100/api/models/';
        const fetchConfig = {
            method: "post",
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json',
            }
        }
        const response = await fetch(vehicleModelUrl, fetchConfig);
        if (response.ok) {
            const newVehicleModel = await response.json();

            setModelName('');
            setPictureUrl('');
            setSelectedManufacturer('');
        }
    };

    async function loadManufacturers() {
        const manufacturersUrl = 'http://localhost:8100/api/manufacturers/';
        const response = await fetch(manufacturersUrl);
        if (response.ok) {
            const data = await response.json();
            setManufacturers(data.manufacturers);
        } else {
            console.error(response);
        }
    }

    useEffect(() => {
        loadManufacturers();
    }, []);



    return (
        <div className="row">
            <div className="offset-3 col-6">
                <div className="shadow p-4 mt-4">
                    <h1>Create a vehicle model</h1>
                    <form onSubmit={handleSubmit} id="create-vehiclemodel-form">
                        <div className="form-floating mb-3">
                            <input onChange={handleModelNameChange} value={modelName} placeholder="Model name"
                                required
                                type="text"
                                name="model_name"
                                id="model_name"
                                className="form-control" />
                            <label htmlFor="model_name">Model Name</label>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handlePictureUrlChange} value={pictureUrl} placeholder="Picture URL"
                                required
                                type="text"
                                name="picture_url"
                                id="picture_url"
                                className="form-control" />
                            <label htmlFor="picture_url">Picture URL</label>
                        </div>
                        <div className="mb-3">
                            <select onChange={handleSelectedManufacturerChange} value={selectedManufacturer} required name="manufacturer" id="manufacturer" className="form-select">
                                <option value="">
                                    Choose a manufacturer
                                </option>
                                {manufacturers.map(manufacturer => {
                                    return (
                                        <option key={manufacturer.id} value={manufacturer.id}>
                                            {manufacturer.name}
                                        </option>
                                    )
                                })}
                            </select>
                        </div>
                        <button className="btn btn-primary">Create</button>
                    </form>
                </div>
            </div>
        </div>
    )
}

export default VehicleModelForm
