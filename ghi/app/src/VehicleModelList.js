import { useState, useEffect } from 'react';

function VehicleModelList() {

    const [vehicleModel, setVehicleModel] = useState([]);

    async function loadVehicleModel() {

        const response = await fetch('http://localhost:8100/api/models/');

        if (response.ok) {
            const data = await response.json();
            setVehicleModel(data.models);
        } else {
            console.error(response);
        }
    };


    const handleDelete = async (id) => {
        const deleteUrl = `http://localhost:8100/api/models/${id}/`

        const fetchConfig = {
            method: "delete",
            headers: {
                'Content-Type': 'application/json',
            }
        }
        const response = await fetch(deleteUrl, fetchConfig)
        if (response.ok) {
            window.location.reload(false)
        }
    }

    useEffect(() => {
        loadVehicleModel();
    }, []);


    return (
        <table className="table table-striped">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Manufacturer</th>
                    <th>Picture</th>
                </tr>
            </thead>
            <tbody>
                {vehicleModel.map(model => {
                    return (
                        <tr key={model.id}>
                            <td>{model.name}</td>
                            <td>{model.manufacturer.name}</td>
                            <td><img src={model.picture_url} className="card-img-top" /></td>
                            <td><button onClick={() => handleDelete(model.id)}>Delete</button></td>
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}

export default VehicleModelList;
