import {Helmet} from "react-helmet-async";

export default function SearchEngineOptimization(props){
    return(
        <>
        <Helmet>
            <title>{props.title}</title>
            <meta name="description" content={props.description}/>
        </Helmet>
        </>
    )
}