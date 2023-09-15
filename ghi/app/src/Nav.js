import { NavLink } from "react-router-dom";
import { Navbar, Nav } from 'react-bootstrap';
import { NavDropdown } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';



function Navigation() {

	const noActiveStyle = {
		backgroundColor: "transparent",
		color: "black",
	};

	const navStyle = {
		backgroundColor: "",
		color: "white",
		margin: 10,
	};

	return (
		<Navbar bg="success" expand="lg" variant="dark">
			<Navbar.Brand as={NavLink} to="/" style={navStyle}>
				Car.ai
			</Navbar.Brand>
			<Navbar.Toggle aria-controls="navbarSupportedContent" />
			<Navbar.Collapse id="navbarSupportedContent">
				<Nav className="me-auto mb-2 mb-lg-0">
					<NavDropdown title="Manufacturers" id="manufacturers-dropdown">
						<NavDropdown.Item as={NavLink} to="/manufacturers" style={noActiveStyle}>
							Manufacturers
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/manufacturers/create" style={noActiveStyle}>
							Create a Manufacturer
						</NavDropdown.Item>
					</NavDropdown>
					<NavDropdown title="Models" id="models-dropdown">
						<NavDropdown.Item as={NavLink} to="/models" style={noActiveStyle}>
							Models
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/models/create" style={noActiveStyle}>
							Create a Model
						</NavDropdown.Item>
					</NavDropdown>
					<NavDropdown title="Automobiles" id="automobiles-dropdown">
						<NavDropdown.Item as={NavLink} to="/automobiles" style={noActiveStyle}>
							Automobiles
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/automobiles/create" style={noActiveStyle}>
							Create an Automobile
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/automobiles/price" style={noActiveStyle}>
							Price an Automobile
						</NavDropdown.Item>
					</NavDropdown>
					<NavDropdown title="Salespeople" id="salespeople-dropdown">
						<NavDropdown.Item as={NavLink} to="/salespeople" style={noActiveStyle}>
							Salespeople
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/salespeople/create" style={noActiveStyle}>
							Add a Salesperson
						</NavDropdown.Item>
					</NavDropdown>
					<NavDropdown title="Customers" id="customers-dropdown">
						<NavDropdown.Item as={NavLink} to="/customers" style={noActiveStyle}>
							Customers
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/customers/create" style={noActiveStyle}>
							Add a Customer
						</NavDropdown.Item>
					</NavDropdown>
					<NavDropdown title="Sales" id="sales-dropdown">
						<NavDropdown.Item as={NavLink} to="/sales" style={noActiveStyle}>
							Sales
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/sales/create" style={noActiveStyle}>
							Add a Sale
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/sales/history" style={noActiveStyle}>
							Salesperson History
						</NavDropdown.Item>
					</NavDropdown>
					<NavDropdown title="Technicians" id="technicians-dropdown">
						<NavDropdown.Item as={NavLink} to="/technicians" style={noActiveStyle}>
							Technicians
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/technicians/create" style={noActiveStyle}>
							Add a Technician
						</NavDropdown.Item>
					</NavDropdown>
					<NavDropdown title="Service Appointments" id="appointments-dropdown">
						<NavDropdown.Item as={NavLink} to="/appointments" style={noActiveStyle}>
							Service Appointments
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/appointments/create" style={noActiveStyle}>
							Add an Appointment
						</NavDropdown.Item>
						<NavDropdown.Item as={NavLink} to="/appointments/history" style={noActiveStyle}>
							Service History
						</NavDropdown.Item>
					</NavDropdown>
					<Nav.Link as={NavLink} to="/dashboard">
						Dashboard
					</Nav.Link>
				</Nav>
			</Navbar.Collapse>
		</Navbar>
	);
}

export default Navigation;
