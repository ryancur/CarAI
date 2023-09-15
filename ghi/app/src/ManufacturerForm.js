import React, { useState } from 'react';

function ManufacturerForm() {

    // initialize variables with state
    const [manufacturerName, setManufacturerName] = useState('');

    // event handlers when changes are made to the form
    const handleManufacturerNameChange = (event) => {
        const value = event.target.value;
        setManufacturerName(value);
    };

    // handle the data once submitted 
    const handleSubmit = async (event) => {
        event.preventDefault();
        const data = {};

        data.name = manufacturerName;

        const manufacturerUrl = 'http://localhost:8100/api/manufacturers/';
        const fetchConfig = {
            method: "post",
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json',
            }
        };
        const response = await fetch(manufacturerUrl, fetchConfig);
        if (response.ok) {
            setManufacturerName('');
        };
    };


    return (
        <div className="row">
            <div className="offset-3 col-6">
                <div className="shadow p-4 mt-4">
                    <h1>Create a Manufacturer</h1>
                    <form onSubmit={handleSubmit} id="create-manufacturer-form">
                        <div className="form-floating mb-3">
                            <input onChange={handleManufacturerNameChange} value={manufacturerName} placeholder="Manufacturer name..." required type="text" name="name" id="name" className="form-control" />
                            <label htmlFor="name">Manufacturer Name</label>
                        </div>
                        <button className="btn btn-primary">Create</button>
                    </form>
                </div>
            </div>
        </div>
    );
}

export default ManufacturerForm;