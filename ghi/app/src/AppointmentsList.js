import { useState, useEffect } from "react";

function AppointmentsList() {
    const [appointments, setAppointments] = useState([]);

    async function loadAppointments() {
        const response = await fetch("http://localhost:8080/api/appointments/");

        if (response.ok) {
            const data = await response.json();

            const newArray = data.appointments.filter(function (element) {
                return element.status === "CREATED";
            });

            setAppointments(newArray);
        } else {
            console.error(response);
        }
    }

    const handleCancelSubmit = async (id) => {
        const cancelUrl = `http://localhost:8080/api/appointments/${id}/cancel/`;

        const fetchConfig = {
            method: "put",
            headers: {
                "Content-Type": "application/json",
            },
        };
        const response = await fetch(cancelUrl, fetchConfig);
        if (response.ok) {
            loadAppointments();
        }
    };

    const handleFinishSubmit = async (id) => {
        const finishUrl = `http://localhost:8080/api/appointments/${id}/finish/`;

        const fetchConfig = {
            method: "put",
            headers: {
                "Content-Type": "application/json",
            },
        };
        const response = await fetch(finishUrl, fetchConfig);
        if (response.ok) {
            loadAppointments();
        }
    };

    useEffect(() => {
        loadAppointments();
    }, []);

    return (
        <div>
            <h1>Service Appointments</h1>
            <table className="table table-striped">
                <thead>
                    <tr>
                        <th>VIN</th>
                        <th>Is VIP?</th>
                        <th>Customer</th>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Technician</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {appointments.map((appt) => {
                        const date = appt.date_time.split("T")[0];
                        const rawTime = appt.date_time.split("T")[1];
                        const time = rawTime.split("+")[0];
                        return (
                            <tr key={appt.id}>
                                <td>{appt.vin}</td>
                                <td>{appt.vip_status ? "Yes" : "No"}</td>
                                <td>{appt.customer}</td>
                                <td>{date}</td>
                                <td>{time}</td>
                                <td>
                                    {appt.technician.first_name + " " + appt.technician.last_name}
                                </td>
                                <td>{appt.reason}</td>
                                <td>
                                    <button onClick={() => handleCancelSubmit(appt.id)}>
                                        Cancel
                                    </button>
                                </td>
                                <td>
                                    <button onClick={() => handleFinishSubmit(appt.id)}>
                                        Finish
                                    </button>
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default AppointmentsList;
