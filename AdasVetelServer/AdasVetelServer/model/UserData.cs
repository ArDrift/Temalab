using System;
using System.Data.Linq.Mapping;

namespace AdasVetelServer.model
{
    [Table(Name = "dbo.UserData")]
    public class UserData : DbElement
    {
        public UserData(string username, string password, int authority)
        {
            Username = username;
            Password = password;
            Auth = authority;
        }

        public UserData() { }

        [Column(IsPrimaryKey = true, Name = "id", IsDbGenerated = true, DbType = "Int NOT NULL IDENTITY")]
        public int Id { get { return az; } set { az = value; } }

        [Column(Name = "time_of_creation")]
        public DateTime? TimeOfCreation { get { return letrehozasdatum; } set { letrehozasdatum = value; } }

        [Column(Name = "username")]
        public string Username { get; set; } = "";


        [Column(Name = "password")]
        public string Password { get; set; } = "";

        [Column(Name = "authority")]
        public int Auth { get; set; } = 0;
    }
}
