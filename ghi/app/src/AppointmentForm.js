import React, { useEffect, useState } from 'react';

function AppointmentForm() {

    // initialize technicians list 
    const [technicians, setTechnicians] = useState([]);

    // initialize variables with state
    const [vin, setVin] = useState('');
    const [customer, setCustomer] = useState('');
    const [apptDate, setApptDate] = useState('');
    const [apptTime, setApptTime] = useState('');
    const [selectedTechnician, setSelectedTechnician] = useState('');
    const [reason, setReason] = useState('');


    // event handlers when changes are made to the form
    const handleVinChange = (event) => {
        const value = event.target.value;
        setVin(value);
    };

    const handleCustomerChange = (event) => {
        const value = event.target.value;
        setCustomer(value);
    };

    const handleApptDateChange = (event) => {
        const value = event.target.value;
        setApptDate(value);
    };

    const handleApptTimeChange = (event) => {
        const value = event.target.value;
        setApptTime(value);
    };

    const handleSelectedTechnicianChange = (event) => {
        const value = event.target.value;
        setSelectedTechnician(value);
    };

    const handleReasonChange = (event) => {
        const value = event.target.value;
        setReason(value);
    };

    // handle the data once submitted 
    const handleSubmit = async (event) => {
        event.preventDefault();
        const data = {};

        data.vin = vin;
        data.customer = customer;
        data.date_time = apptDate + " " + apptTime;
        data.technician = selectedTechnician;
        data.reason = reason;

        const appointmentUrl = 'http://localhost:8080/api/appointments/';
        const fetchConfig = {
            method: "post",
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json',
            }
        };

        const response = await fetch(appointmentUrl, fetchConfig);

        if (response.ok) {

            setVin('');
            setCustomer('');
            setApptDate('');
            setApptTime('');
            setSelectedTechnician('');
            setReason('');
        };
    };

    // fetch technician data to select a technician in the form
    const fetchData = async () => {
        const techniciansUrl = 'http://localhost:8080/api/technicians/';
        const response = await fetch(techniciansUrl);
        if (response.ok) {
            const data = await response.json();
            setTechnicians(data.technicians);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);


    return (
        <div className="row">
            <div className="offset-3 col-6">
                <div className="shadow p-4 mt-4">
                    <h1>Create a Service Appointment</h1>
                    <form onSubmit={handleSubmit} id="create-technician-form">
                        <div className="form-floating mb-3">
                            <input onChange={handleVinChange} value={vin} placeholder="VIN..." required type="text" name="vin" id="vin" className="form-control" />
                            <label htmlFor="vin">VIN</label>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handleCustomerChange} value={customer} placeholder="Full Name..." required type="text" name="customer" id="customer" className="form-control" />
                            <label htmlFor="customer">Customer</label>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handleApptDateChange} value={apptDate} placeholder="yyyy-mm-dd" type="date" name="appt_date" id="appt_date" className="form-control" />
                            <label htmlFor="appt_date">Date</label>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handleApptTimeChange} value={apptTime} placeholder="--:-- --" type="time" name="appt_time" id="appt_time" className="form-control" />
                            <label htmlFor="appt_time">Time</label>
                        </div>
                        <div className="mb-3">
                            <label htmlFor="technician">Technician</label>
                            <select onChange={handleSelectedTechnicianChange} value={selectedTechnician} required name="technician" id="technician" className="form-control">
                                <option value="">Choose a technician...</option>
                                {technicians.map(tech => {
                                    return (
                                        <option key={tech.id} value={tech.id}>
                                            {tech.first_name + " " + tech.last_name}
                                        </option>
                                    )
                                })}
                            </select>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handleReasonChange} value={reason} placeholder="" type="text" name="reason" id="reason" className="form-control" />
                            <label htmlFor="reason">Reason</label>
                        </div>
                        <button className="btn btn-primary">Create</button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default AppointmentForm;