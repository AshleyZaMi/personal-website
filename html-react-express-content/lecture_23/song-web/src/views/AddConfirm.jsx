import { Link, useLocation } from "react-router-dom"

export default function AddConfirm() {
	const location = useLocation()
	return (
		<>
			<ol className="breadcrumb p-3">
				<li className="breadcrumb-item"><Link to="/">Home</Link></li>
				<li className="breadcrumb-item"><Link to="/addForm">Add</Link></li>
				<li className="breadcrumb-item active">Confirmation</li>
			</ol>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4">Add a Song</h1>
				</div>
			</div>
			<div className="container">
				<div className="row mt-4">
					<div className="col-12">

						{location.state.status == 201 ? (
							<div className="text-success">
								<span className="font-italic">{location.state.trackName}</span> was successfully added.
							</div>
						) : (
							<div className="text-danger">
								<span className="font-italic">{location.state.trackName}</span> was not added.
								<p>Error: {location.state.message}</p>
							</div>
						)}

					</div>
				</div>
				<div className="row mt-4 mb-4">
					<div className="col-12">
						<Link to="/addForm" role="button" className="btn btn-primary">Back to Add Form</Link>
					</div>
				</div>
			</div>
		</>
	)
}

