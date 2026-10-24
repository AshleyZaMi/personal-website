import { Link, useLocation } from "react-router-dom"

export default function EditConfirmation() {
	const location = useLocation()

	return (
		<>
			<ol className="breadcrumb p-3">
				<li className="breadcrumb-item"><Link to="/">Home</Link></li>
				<li className="breadcrumb-item"><Link to="/searchForm">Search</Link></li>
				<li className="breadcrumb-item"><Link to="/searchResults">Results</Link></li>
				<li className="breadcrumb-item active">Confirmation</li>
			</ol>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4">Edit a Song</h1>
				</div>
			</div>
			<div className="container">
				<div className="row mt-4">
					<div className="col-12">

						{
							location.state.status == 200 ?
								<div className="text-success">
								<span className="font-italic">{location.state.trackName}</span> was successfully updated.
							</div>
							:
							<div className="text-error">
							<span className="font-italic">{location.state.trackName}</span> was not updated.
							<p>Error: {location.state.message}</p>
							</div>
						}
						{/* // <div className="text-success">
						// 	<span className="font-italic">Track Name</span> was successfully updated.
						// </div> */}

					</div>
				</div>
				<div className="row mt-4 mb-4">
					<div className="col-12">
						<Link to="/searchForm" role="button" className="btn btn-primary">Back to Search Form</Link>
					</div>
				</div>
			</div>
		</>
	)
}
