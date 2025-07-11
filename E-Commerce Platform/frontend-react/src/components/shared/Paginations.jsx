import { Pagination } from "@mui/material";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";


const Paginations = ({numberofPage, totalProducts}) => {
    const [searchParams] = useSearchParams();
    const pathname = useLocation().pathname;
    const params = new URLSearchParams(searchParams);
    const navigate = useNavigate();
    const paraValue = searchParams.get("page")
            ? Number(searchParams.get("page"))
            : 1;
    
    const onChangeHandler = (event, value) => {
        params.set("page", value.toString());
        navigate(`${pathname}?${params}`)
    };


    return (
        <div>
            <Pagination
            count={numberofPage} 
            page={paraValue}
            defaultPage={1} 
            siblingCount={1} 
            boundaryCount={2}
            onChange={onChangeHandler}
            />
        </div>

    )
};

export default Paginations;