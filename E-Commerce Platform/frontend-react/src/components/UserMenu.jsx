import { Avatar } from '@mui/material';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import React from "react";
import { useDispatch, useSelector } from 'react-redux';
import { Link, useNavigate } from 'react-router-dom';
import { BiUser } from 'react-icons/bi';
import { FaShoppingCart } from 'react-icons/fa';
import { IoExitOutline } from 'react-icons/io5';
import BackDrop from './BackDrop';
import { logoutUser } from '../store/action';

const UserMenu = () => {
    const [anchorEl, setAnchorEl] = React.useState(null);
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const open = Boolean(anchorEl);
    const handleClick = (event) => {
        setAnchorEl(event.currentTarget);
    };
    const handleClose = () => {
        setAnchorEl(null);
    };
    const { user } = useSelector((state)=> state.auth);

    const logoutHandler = () => {
        dispatch(logoutUser(navigate));
    };

    return (
        <div className='relative z-30'>
            <div
            className='sm:border-[1px] sm:border-slate-400 flex flex-col items-center gap-1 rounded-full cursor-pointer hover:shadow-md transition text-slate-700'
                onClick={handleClick}
            >
            <Avatar alt='Menu' />
            </div>
        <Menu
            sx={{width:"400px"}}
            id="basic-menu"
            anchorEl={anchorEl}
            open={open}
            onClose={handleClose}
            slotProps={{
            list: {
                'aria-labelledby': 'basic-button',
                sx:{width:160},
            },
            }}
        >
            <Link to="/profile">
                <MenuItem className='flex gap-2' 
                onClick={handleClose}>
                    <BiUser className="text-xl" />
                    <span className='font-bold text-[16px] mt-1'>
                        {user?.username}
                    </span>
                </MenuItem>            
            </Link>


            <Link to="/profile/orders">
                <MenuItem className='flex gap-2' 
                onClick={handleClose}>
                    <FaShoppingCart className="text-xl" />
                    <span className='font-semibold'>
                        Order
                    </span>
                </MenuItem>            
            </Link>            

       
            <MenuItem className='flex gap-2' 
            onClick={logoutHandler}>
                <div className="flex items-center space-x-2 px-4 py-[6px] 
                            bg-gradient-to-r from-purple-600 to-red-500 
                            text-white font-semibold rounded-md shadow-lg 
                            hover:from-purple-500 hover:to-red-400 transition 
                            duration-300 ease-in-out transform ">
                <IoExitOutline className="text-xl" />
                <span className='font-bold text-[16px] mt-1'>
                    LogOut
                </span>
                </div>
            </MenuItem>            
        </Menu>


        { open && <BackDrop />}

        



        </div>
    );
}


export default UserMenu;